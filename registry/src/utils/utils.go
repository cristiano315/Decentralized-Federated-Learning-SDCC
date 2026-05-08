package utils

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"

	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	dbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"

	"github.com/aws/aws-sdk-go-v2/service/ecs"
	ecstypes "github.com/aws/aws-sdk-go-v2/service/ecs/types"

	pb "federate-registry/federated"
)

func AddNode(node *pb.NodeInfo) error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		return err
	}

	// Using the Config value, create the DynamoDB client
	ddb_client := dynamodb.NewFromConfig(cfg)

	item, err := attributevalue.MarshalMap(node)
	if err != nil {
		return err
	}

	_, err = ddb_client.PutItem(context.TODO(), &dynamodb.PutItemInput{
		TableName: aws.String("Nodes"),
		Item:      item,
	})
	return err
}

func RemoveNode(nodeId string) error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		return err
	}

	// Using the Config value, create the DynamoDB client
	ddb_client := dynamodb.NewFromConfig(cfg)

	key, err := attributevalue.MarshalMap(map[string]string{
		"NodeId": nodeId,
	})

	// Define the DeleteItem input
	input := &dynamodb.DeleteItemInput{
		TableName: aws.String("Nodes"),
		Key:       key,
	}

	// Execute the request
	_, err = ddb_client.DeleteItem(context.TODO(), input)
	return err
}

func NodeCount() (int32, error) {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		return 0, err
	}

	// Using the Config value, create the DynamoDB client
	ddb_client := dynamodb.NewFromConfig(cfg)

	var count int32

	paginator := dynamodb.NewScanPaginator(ddb_client, &dynamodb.ScanInput{
		TableName: aws.String("Nodes"),
		Select:    dbtypes.SelectCount,
	})

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(context.TODO())
		if err != nil {
			return 0, err
		}
		count += page.Count
	}

	return count, nil
}

// TODO: Add IDLE and WORKING
func FetchActiveNodes() ([]*pb.NodeInfo, error) {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		return nil, err
	}

	// Using the Config value, create the DynamoDB client
	ddb_client := dynamodb.NewFromConfig(cfg)

	var items []*pb.NodeInfo

	// Create the paginator
	paginator := dynamodb.NewScanPaginator(ddb_client, &dynamodb.ScanInput{
		TableName: aws.String("Nodes"),
	})

	// Iterate through pages
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(context.TODO())
		if err != nil {
			fmt.Printf("Failed to retrive page  %v\n", err)
		}

		for _, itemMap := range page.Items {
			var client *pb.NodeInfo

			// Unmarshal a single item
			err := attributevalue.UnmarshalMap(itemMap, &client)
			if err != nil {
				fmt.Printf("Failed to unmarshal: %v\n", err)
			}

			items = append(items, client)
		}
	}
	return items, err
}

func LaunchTask(task string, ammount int32) error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		return err
	}

	// Using the Config value, create the ECS client
	client := ecs.NewFromConfig(cfg)

	// Define the RunTask input
	input := &ecs.RunTaskInput{
		Cluster:        aws.String("clients"),
		TaskDefinition: aws.String(task),
		LaunchType:     ecstypes.LaunchTypeFargate,
		Count:          aws.Int32(ammount),
		/*
					Overrides: &ecstypes.TaskOverride{
			            ContainerOverrides: []ecstypes.ContainerOverride{
			                {
			                    Name: aws.String("my-app-container"), // Must match Task Def
			                    Environment: []ecstypes.KeyValuePair{
			                        {
			                            Name:  aws.String("STAGE"),
			                            Value: aws.String("PROD"),
			                        },
			                    },
			                },
			            },
			        },
		*/

		NetworkConfiguration: &ecstypes.NetworkConfiguration{
			AwsvpcConfiguration: &ecstypes.AwsVpcConfiguration{
				Subnets: []string{"subnet-04c8bd531ed80fa30", "subnet-02d82ec3a388354c1", "subnet-0569bbf51b6235702", "subnet-0caba6f1751703166", "subnet-0f9fb6c71c73e8a44", "subnet-0fafc3fd7c6512117"},
				//SecurityGroups: []string{"sg-zzzzzzzz"},
				AssignPublicIp: ecstypes.AssignPublicIpEnabled,
			},
		},
	}

	// Execute the request
	output, err := client.RunTask(context.TODO(), input)
	if err != nil {
		return err
	}

	for _, task := range output.Tasks {
		log.Printf("Task started! ARN: %s", *task.TaskArn)
	}

	return err
}

// NOT TESTED YET
func DestroyTask(task string) error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		return err
	}

	// Using the Config value, create the ECS client
	ecs_client := ecs.NewFromConfig(cfg)

	// Define the StopTask input
	input := &ecs.StopTaskInput{
		Cluster: aws.String("clients"),
		Task:    aws.String(task),
		Reason:  aws.String("Manually stopped via Go SDK"),
	}

	// Execute the request
	_, err = ecs_client.StopTask(context.TODO(), input)
	return err
}

func ScaleService(ammount int32) error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		return err
	}

	// Using the Config value, create the ECS client
	ecs_client := ecs.NewFromConfig(cfg)

	// Define the UpdateService input
	input := &ecs.UpdateServiceInput{
		Cluster:      aws.String("clients"),
		Service:      aws.String("test_client-service"),
		DesiredCount: aws.Int32(ammount),
	}

	// Execute the request
	_, err = ecs_client.UpdateService(context.TODO(), input)
	if err != nil {
		return err
	}

	// Initialize the Waiter (15 seconds by default)
	waiter := ecs.NewServicesStableWaiter(ecs_client)

	// Define max waiting time
	maxWaitTime := 2 * time.Minute

	err = waiter.Wait(context.TODO(), &ecs.DescribeServicesInput{
		Cluster:  aws.String("clients"),
		Services: []string{"test_client-service"},
	}, maxWaitTime)
	return err
}

// NOT TESTED YET
func KillAllTasks() error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		return err
	}

	// Using the Config value, create the ECS client
	ecs_client := ecs.NewFromConfig(cfg)

	// Get all Task ARNs
	listOutput, err := ecs_client.ListTasks(context.TODO(), &ecs.ListTasksInput{
		Cluster:     aws.String("clients"),
		ServiceName: aws.String("test_client-service"),
	})
	if err != nil {
		return err
	}

	for _, taskArn := range listOutput.TaskArns {
		_, err := ecs_client.StopTask(context.TODO(), &ecs.StopTaskInput{
			Cluster: aws.String("clients"),
			Task:    aws.String(taskArn),
			Reason:  aws.String("Mass termination triggered by Go SDK"),
		})
		if err != nil {
			fmt.Printf("Failed to stop %s: %v\n", taskArn, err)
		}
	}

	return err
}
