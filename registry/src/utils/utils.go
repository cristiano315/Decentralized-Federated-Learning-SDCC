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

	pb "federate-registry/federated"
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
				Subnets: []string{"subnet-049b938577203e317", "subnet-04ec6d7a5a15cedcc", "subnet-06db11df621ee60eb", "subnet-0ac28f8e5d64980c9", "subnet-0c126c2d6158e8ccd", "subnet-08394add4030de621"},
				//SecurityGroups: []string{"sg-xxxxxxxx"},
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

func GetLocalIP() (string, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}

	for _, address := range addrs {
		if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			// Prendi solo l'IPv4
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
