package utils

import (
	"context"
	"fmt"
	"log"

	"net"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"

	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	dbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"

	"github.com/aws/aws-sdk-go-v2/service/ecs"
	ecstypes "github.com/aws/aws-sdk-go-v2/service/ecs/types"

	pb "federate-registry-centralized/federated"
)

// =====================================================================
// ENVIROMENT VARIABLE STRUCT
// =====================================================================

type EnvVar struct {
	Key   string
	Value string
}

// =====================================================================
// FUNCTIONS
// =====================================================================

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
		TableName: aws.String("Nodes_Centralized"),
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
		TableName: aws.String("Nodes_Centralized"),
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
		TableName: aws.String("Nodes_Centralized"),
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

func FetchIdleNodes() ([]*pb.NodeInfo, error) {
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
		TableName:        aws.String("Nodes_Centralized"),
		FilterExpression: aws.String("#st = :statusVal"),
		ExpressionAttributeNames: map[string]string{
			"#st": "Status",
		},
		ExpressionAttributeValues: map[string]dbtypes.AttributeValue{
			":statusVal": &dbtypes.AttributeValueMemberS{Value: "idle"},
		},
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

func ChangeStatus(nodeId string, new_status string) error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		return err
	}

	// Using the Config value, create the DynamoDB client
	ddb_client := dynamodb.NewFromConfig(cfg)

	// Define the UpdateItem input
	input := &dynamodb.UpdateItemInput{
		TableName: aws.String("Nodes_Centralized"),
		// Identify the specific row by its Primary Key
		Key: map[string]dbtypes.AttributeValue{
			"NodeId": &dbtypes.AttributeValueMemberS{Value: nodeId},
		},
		UpdateExpression: aws.String("SET #st = :newStatusVal"),
		ExpressionAttributeNames: map[string]string{
			"#st": "Status",
		},
		ExpressionAttributeValues: map[string]dbtypes.AttributeValue{
			":newStatusVal": &dbtypes.AttributeValueMemberS{Value: new_status},
		},
	}

	// Execute the request
	_, err = ddb_client.UpdateItem(context.TODO(), input)
	if err != nil {
		return fmt.Errorf("Failed to update node status: %w", err)
	}
	return nil
}

func LaunchTask(cluster string, task string, ammount int32, container string, params []EnvVar) error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		return err
	}

	// Using the Config value, create the ECS client
	client := ecs.NewFromConfig(cfg)

	envVariables := make([]ecstypes.KeyValuePair, len(params))

	for i, p := range params {
		envVariables[i] = ecstypes.KeyValuePair{
			Name:  aws.String(p.Key),
			Value: aws.String(p.Value),
		}
	}

	// Define the RunTask input
	input := &ecs.RunTaskInput{
		Cluster:        aws.String(cluster),
		TaskDefinition: aws.String(task),
		LaunchType:     ecstypes.LaunchTypeFargate,
		Count:          aws.Int32(ammount),

		NetworkConfiguration: &ecstypes.NetworkConfiguration{
			AwsvpcConfiguration: &ecstypes.AwsVpcConfiguration{
				Subnets:        []string{"subnet-0461b7700dadc71ea", "subnet-0b69b19f072ee6070", "subnet-0932e497e7f35ba86", "subnet-04a2749a19f244591", "subnet-0f202eb907c098dd2", "subnet-subnet-0a6ffc73ca4b2b632"},
				AssignPublicIp: ecstypes.AssignPublicIpEnabled,
			},
		},

		Overrides: &ecstypes.TaskOverride{
			ContainerOverrides: []ecstypes.ContainerOverride{
				{
					Name:        aws.String(container),
					Environment: envVariables,
				},
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

func GetLocalIP() (string, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}

	for _, address := range addrs {
		if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			// skip link-local addresses like 169.254.x.x
			if ipnet.IP.IsLinkLocalUnicast() {
				continue
			}

			// return the first valid IPv4
			if ipnet.IP.To4() != nil {
				return ipnet.IP.String(), nil
			}
		}
	}
	return "", fmt.Errorf("nessun IP valido trovato")
}

func GetPort() string {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	return port
}

func GetFullLocalAdress() string {
	ip, _ := GetLocalIP()
	port := GetPort()

	return net.JoinHostPort(ip, port)
}
