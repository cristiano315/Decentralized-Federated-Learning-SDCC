package utils

import (
	"context"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"

	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"

	pb "federate-registry/federated"
)

/*
type Node struct {
	node_id    string `dynamodbav:"node_id"`
	ip_address string `dynamodbav:"ip_address"`
	port       int    `dynamodbav:"port"`
}
*/

func AddNode(node *pb.NodeInfo) error {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		log.Fatalf("unable to load SDK config, %v", err)
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

/*
func GetUser(ctx context.Context, client *dynamodb.Client, name string) (*User, error) {
	key, err := attributevalue.MarshalMap(map[string]string{
		"Name": "TestGo",
		"Id":   "0",
	})
	if err != nil {
		log.Fatalf("failed to marshal key: %v", err)
	}

	output, err := client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName: aws.String("serviceRegistry"),
		Key:       key,
	})
	if err != nil {
		log.Fatalf("failed to get item: %v", err)
	}
	if output.Item == nil {
		fmt.Println("No item found with that ID")
		return nil, err
	}

	var user User
	err = attributevalue.UnmarshalMap(output.Item, &user)
	if err != nil {
		log.Fatalf("failed to unmarshal item: %v", err)
	}
	return &user, err
}

func ListAllServices(ctx context.Context, client *ecs.Client, cluster string) {
	// 1. Initialize the paginator
	paginator := ecs.NewListServicesPaginator(client, &ecs.ListServicesInput{
		Cluster: aws.String(cluster),
	})

	fmt.Println("Services in cluster:")

	// 2. Iterate through all pages
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			panic(err)
		}

		for _, serviceArn := range page.ServiceArns {
			fmt.Printf(" - %s\n", serviceArn)
		}
	}
}

func ExecuteTask(ctx context.Context, client *ecs.Client, cluster string, taskDef string, ammount int32) (*ecs.RunTaskOutput, error) {

	input := &ecs.RunTaskInput{
		Cluster:        aws.String(cluster),
		TaskDefinition: aws.String(taskDef),
		LaunchType:     types.LaunchTypeFargate,
		Count:          aws.Int32(ammount),
		NetworkConfiguration: &types.NetworkConfiguration{
			AwsvpcConfiguration: &types.AwsVpcConfiguration{
				Subnets: []string{"subnet-04c8bd531ed80fa30", "subnet-02d82ec3a388354c1", "subnet-0569bbf51b6235702", "subnet-0caba6f1751703166", "subnet-0f9fb6c71c73e8a44", "subnet-0fafc3fd7c6512117"},
				//SecurityGroups: []string{"sg-zzzzzzzz"},
				AssignPublicIp: types.AssignPublicIpEnabled,
			},
		},
	}

	output, err := client.RunTask(ctx, input)
	if err != nil {
		log.Fatalf("failed to run task: %v", err)
	}

	for _, task := range output.Tasks {
		log.Printf("Task started! ARN: %s", *task.TaskArn)
	}

	return output, err
}

func main() {
	// Using the SDK's default configuration, load additional config
	// and credentials values from the environment variables, shared
	// credentials, and shared configuration files
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		log.Fatalf("unable to load SDK config, %v", err)
	}

	// Using the Config value, create the DynamoDB client
	ddb_client := dynamodb.NewFromConfig(cfg)

	// Build the request with its input parameters
	resp, err := ddb_client.ListTables(context.TODO(), &dynamodb.ListTablesInput{
		Limit: aws.Int32(5),
	})
	if err != nil {
		log.Fatalf("failed to list tables, %v", err)
	}

	fmt.Println("Tables:")
	for _, tableName := range resp.TableNames {
		fmt.Println(tableName)
	}

	newUser := User{
		Name:   "TestGo",
		Id:     "0",
		Adress: "00.0.0.00",
	}

	fmt.Printf("Adding: %s...\n", newUser.Name)
	err = AddUser(context.TODO(), ddb_client, newUser)
	if err != nil {
		log.Fatalf("failed to add user: %v", err)
	}

	fmt.Println("Retrieving user...")
	retrievedUser, err := GetUser(context.TODO(), ddb_client, "TestGo")
	if err != nil {
		log.Fatalf("failed to get user: %v", err)
	}

	if retrievedUser != nil {
		fmt.Printf("Found User: %+v\n", retrievedUser)
	} else {
		fmt.Println("User not found.")
	}

	// Using the Config value, create the ECS client
	ecs_client := ecs.NewFromConfig(cfg)

	ListAllServices(context.TODO(), ecs_client, "clients")

	output, err := ExecuteTask(context.TODO(), ecs_client, "clients", "test_client", 3)
	if output != nil {
		fmt.Printf("Output: %+v\n", output)
	} else {
		fmt.Println("Error while getting output.")
	}

}
*/
