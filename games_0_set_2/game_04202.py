import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to change direction. Survive and score points by eating food (red dots). "
        "Get bonus points for moving close to obstacles (grey blocks), but don't hit them!"
    )

    game_description = (
        "A fast-paced, grid-based Snake game. Risky plays near obstacles yield bonus points, but safe moves "
        "incur penalties. Maximize your score before you run out of moves or crash!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
    
    MAX_STEPS = 1000
    WIN_SCORE = 100
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 50)
    COLOR_SNAKE_HEAD = (50, 255, 50)
    COLOR_SNAKE_BODY = (40, 200, 40)
    COLOR_FOOD = (255, 50, 50)
    COLOR_OBSTACLE = (100, 100, 110)
    COLOR_OBSTACLE_FLASH = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)

    # Directions (dx, dy)
    DIR_UP = (0, -1)
    DIR_DOWN = (0, 1)
    DIR_LEFT = (-1, 0)
    DIR_RIGHT = (1, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_body = deque()
        self.snake_direction = self.DIR_RIGHT
        self.food_pos = (0, 0)
        self.obstacles = []
        self.last_event_reward = 0

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize snake
        start_pos = (self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2)
        self.snake_body = deque([start_pos, (start_pos[0] - 1, start_pos[1]), (start_pos[0] - 2, start_pos[1])])
        self.snake_direction = self.DIR_RIGHT
        
        self._spawn_obstacles()
        self._spawn_food()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Store pre-move state for reward calculation ---
        old_head_pos = self.snake_body[0]
        old_dist_to_food = self._manhattan_distance(old_head_pos, self.food_pos)
        old_dist_to_obstacle, _ = self._distance_to_nearest_obstacle(old_head_pos)

        # --- 2. Process action and move snake ---
        self._update_snake_direction(action[0])
        self._move_snake()
        
        # --- 3. Check for events (food, collisions) ---
        self.last_event_reward = 0
        terminated = self._check_events()

        # --- 4. Calculate reward ---
        new_head_pos = self.snake_body[0]
        new_dist_to_food = self._manhattan_distance(new_head_pos, self.food_pos)
        new_dist_to_obstacle, is_adjacent = self._distance_to_nearest_obstacle(new_head_pos)
        
        reward = self._calculate_reward(
            old_dist_to_food, new_dist_to_food, 
            old_dist_to_obstacle, new_dist_to_obstacle,
            is_adjacent
        )

        # --- 5. Update step counter and check for termination ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE:
            terminated = True
        
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                self.last_event_reward = 100 # Win bonus
            else:
                self.last_event_reward = -100 # Lose penalty
            self.game_over = True

        reward += self.last_event_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_snake_direction(self, movement_action):
        new_dir = self.snake_direction
        if movement_action == 1 and self.snake_direction != self.DIR_DOWN: # Up
            new_dir = self.DIR_UP
        elif movement_action == 2 and self.snake_direction != self.DIR_UP: # Down
            new_dir = self.DIR_DOWN
        elif movement_action == 3 and self.snake_direction != self.DIR_RIGHT: # Left
            new_dir = self.DIR_LEFT
        elif movement_action == 4 and self.snake_direction != self.DIR_LEFT: # Right
            new_dir = self.DIR_RIGHT
        # Action 0 (no-op) means continue in the same direction
        self.snake_direction = new_dir

    def _move_snake(self):
        head = self.snake_body[0]
        dx, dy = self.snake_direction
        
        new_head = ((head[0] + dx) % self.GRID_WIDTH, (head[1] + dy) % self.GRID_HEIGHT)
        self.snake_body.appendleft(new_head)
    
    def _check_events(self):
        head = self.snake_body[0]
        
        # Food collision
        if head == self.food_pos:
            self.score += 10
            self.last_event_reward += 10 # Event reward for eating
            # SFX: play('eat_sound.wav')
            self._spawn_food()
        else:
            self.snake_body.pop() # Remove tail if no food eaten
            
        # Self collision
        if head in list(self.snake_body)[1:]:
            # SFX: play('crash_sound.wav')
            return True
            
        # Obstacle collision
        for obs in self.obstacles:
            if obs.collidepoint(head):
                # SFX: play('crash_sound.wav')
                return True
                
        return False

    def _calculate_reward(self, old_dist_food, new_dist_food, old_dist_obs, new_dist_obs, is_adjacent):
        reward = 0.0

        # Reward for moving towards food
        if new_dist_food < old_dist_food:
            reward += 0.1
        # Penalty for moving away from food
        else:
            reward -= 0.2
        
        # Penalty for moving towards an obstacle
        if new_dist_obs < old_dist_obs:
            reward -= 0.5

        # Reward for being adjacent to an obstacle (risk-taking)
        if is_adjacent:
            reward += 1.0

        return reward

    def _get_empty_cell(self):
        occupied_cells = set(self.snake_body)
        for obs in self.obstacles:
            for x in range(obs.left, obs.right):
                for y in range(obs.top, obs.bottom):
                    occupied_cells.add((x, y))
        
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in occupied_cells:
                return pos

    def _spawn_food(self):
        self.food_pos = self._get_empty_cell()

    def _spawn_obstacles(self):
        self.obstacles = []
        num_obstacles = self.np_random.integers(4, 8)
        for _ in range(num_obstacles):
            w = self.np_random.integers(2, 6)
            h = self.np_random.integers(2, 6)
            x = self.np_random.integers(0, self.GRID_WIDTH - w)
            y = self.np_random.integers(0, self.GRID_HEIGHT - h)
            
            # Ensure obstacles don't spawn on the snake's initial path
            new_obs = pygame.Rect(x, y, w, h)
            is_safe = True
            for segment in self.snake_body:
                if new_obs.collidepoint(segment):
                    is_safe = False
                    break
            if is_safe:
                self.obstacles.append(new_obs)

    def _manhattan_distance(self, p1, p2):
        # Grid wrapping distance
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return min(dx, self.GRID_WIDTH - dx) + min(dy, self.GRID_HEIGHT - dy)

    def _distance_to_nearest_obstacle(self, pos):
        min_dist = float('inf')
        is_adjacent = False
        if not self.obstacles:
            return min_dist, is_adjacent

        for obs in self.obstacles:
            # Check for adjacency (Chebyshev distance of 1)
            # Inflate rect by 1 in each direction and check for collision
            adj_rect = obs.inflate(2, 2)
            if adj_rect.collidepoint(pos) and not obs.collidepoint(pos):
                is_adjacent = True

            # Calculate Manhattan distance to the closest point on the obstacle
            closest_x = max(obs.left, min(pos[0], obs.right - 1))
            closest_y = max(obs.top, min(pos[1], obs.bottom - 1))
            dist = self._manhattan_distance(pos, (closest_x, closest_y))
            if dist < min_dist:
                min_dist = dist
        
        return min_dist, is_adjacent

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, 
                             (obs.x * self.CELL_SIZE, obs.y * self.CELL_SIZE, 
                              obs.w * self.CELL_SIZE, obs.h * self.CELL_SIZE))

        # Draw obstacle flash if adjacent
        if self.snake_body: # Check if snake exists
            _, is_adjacent = self._distance_to_nearest_obstacle(self.snake_body[0])
            if is_adjacent:
                # SFX: play('proximity_hum.wav')
                head_rect = pygame.Rect(self.snake_body[0][0] * self.CELL_SIZE, self.snake_body[0][1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.gfxdraw.box(self.screen, head_rect.inflate(8, 8), (*self.COLOR_OBSTACLE_FLASH, 40))

        # Draw food
        food_rect = (int(self.food_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
                     int(self.food_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2))
        pygame.draw.circle(self.screen, self.COLOR_FOOD, food_rect, self.CELL_SIZE // 2)
        pygame.gfxdraw.aacircle(self.screen, food_rect[0], food_rect[1], self.CELL_SIZE // 2 -1, self.COLOR_FOOD)

        # Draw snake
        for i, segment in enumerate(self.snake_body):
            rect = (segment[0] * self.CELL_SIZE, segment[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        steps_text = self.font_large.render(f"MOVES: {self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 5))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            status_render = self.font_large.render(status_text, True, self.COLOR_TEXT)
            status_rect = status_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(status_render, status_rect)

            final_score_render = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_render, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
            "food_pos": self.food_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Call reset() FIRST to initialize the game state. This prevents errors
        # in other validation steps that rely on a valid state (e.g., an initialized snake).
        obs, info = self.reset(seed=42)

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (now that state is initialized)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset again to ensure it's idempotent
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert len(self.snake_body) > 0, "Snake not initialized in reset"
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test food eating
        self.reset(seed=123)
        initial_length = len(self.snake_body)
        initial_score = self.score
        
        # Predict next head position based on current direction
        head = self.snake_body[0]
        dx, dy = self.snake_direction
        next_head_pos = ((head[0] + dx) % self.GRID_WIDTH, (head[1] + dy) % self.GRID_HEIGHT)
        self.food_pos = next_head_pos # Place food where the head will be

        # Step forward without changing direction to guarantee eating
        self.step([0, 0, 0])
        
        assert len(self.snake_body) == initial_length + 1
        assert self.score == initial_score + 10

        self.reset() # Reset back to a clean state

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Override screen for display
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Risk")

    # Game loop
    action = [0, 0, 0] # Start with no-op
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                    action = [0, 0, 0]
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            # print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            action[0] = 0 # Reset movement action after each step for turn-based control

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(10) # Control human play speed

    env.close()