import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro arcade Snake game.

    The player controls a snake on a grid, aiming to eat food pellets.
    Eating a pellet increases the snake's length and the player's score.
    The game ends if the snake hits a wall, collides with its own body,
    reaches the max step limit, or achieves the winning score.

    Rewards are structured to encourage efficient and risky gameplay.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use ↑↓←→ to change the snake's direction. "
        "Try to eat the yellow pellets!"
    )

    # Short, user-facing description of the game
    game_description = (
        "Guide a growing snake to devour glowing food pellets. "
        "Achieve a score of 100 to win. Colliding with yourself or the walls ends the game."
    )

    # The game state is static until an action is received
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_WALL = (80, 80, 100)
    COLOR_SNAKE_HEAD = (100, 255, 100)
    COLOR_SNAKE_BODY = (0, 200, 0)
    COLOR_FOOD = (255, 255, 0)
    COLOR_FOOD_GLOW = (255, 255, 0, 64)
    COLOR_TEXT = (220, 220, 255)
    COLOR_REWARD_POS = (0, 255, 255)
    COLOR_REWARD_NEG = (255, 100, 100)
    
    # Game parameters
    WIN_SCORE = 100
    MAX_STEPS = 1000
    INITIAL_SNAKE_LENGTH = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables
        self.snake_pos = None
        self.direction = None
        self.direction_vec = None
        self.food_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.reward_popups = []

        # Initialize state
        # The reset method is called to set up the initial state.
        # However, some attributes like np_random are set by the parent's reset,
        # so we ensure a seed is available for the first reset.
        self.reset(seed=0)
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_popups = []

        # Center the snake
        start_x = self.GRID_WIDTH // 2
        start_y = self.GRID_HEIGHT // 2
        self.snake_pos = deque(
            [(start_x - i, start_y) for i in range(self.INITIAL_SNAKE_LENGTH)]
        )
        
        self.direction = 4 # 1:UP, 2:DOWN, 3:LEFT, 4:RIGHT
        self.direction_vec = (1, 0) # (dx, dy)

        self._place_food()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self._update_direction(movement)

        # Store state for reward calculation
        old_head = self.snake_pos[0]
        dist_before = math.hypot(old_head[0] - self.food_pos[0], old_head[1] - self.food_pos[1])
        
        # Move snake
        new_head = (self.snake_pos[0][0] + self.direction_vec[0], 
                    self.snake_pos[0][1] + self.direction_vec[1])
        
        # Check for termination conditions
        collided = self._check_collision(new_head)
        if collided:
            self.game_over = True
            reward = -10.0
            self._add_reward_popup(reward, new_head)
            # sfx: game_over_sound
        else:
            self.snake_pos.appendleft(new_head)
            
            # Check for food
            if new_head == self.food_pos:
                self.score += 1
                # sfx: eat_food_sound
                self._place_food()
                # Don't pop tail to grow
                reward = 1.0
            else:
                self.snake_pos.pop()
                reward = 0.0

            # Calculate rewards
            dist_after = math.hypot(new_head[0] - self.food_pos[0], new_head[1] - self.food_pos[1])
            
            # Reward for getting closer to food
            distance_reward = (dist_before - dist_after) * 0.1
            reward += distance_reward

            # Reward/penalty for risk
            if self._is_risky_move(new_head):
                risk_reward = 2.0
                reward += risk_reward
            else:
                risk_penalty = -0.2
                reward += risk_penalty
            
            self._add_reward_popup(reward, new_head)
        
        # Win condition
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            reward += 100.0
            self._add_reward_popup(100, new_head)
            # sfx: win_game_sound

        self.steps += 1
        terminated = self.game_over or (self.steps >= self.MAX_STEPS)
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_direction(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.direction != 2: # UP
            self.direction = 1
            self.direction_vec = (0, -1)
        elif movement == 2 and self.direction != 1: # DOWN
            self.direction = 2
            self.direction_vec = (0, 1)
        elif movement == 3 and self.direction != 4: # LEFT
            self.direction = 3
            self.direction_vec = (-1, 0)
        elif movement == 4 and self.direction != 3: # RIGHT
            self.direction = 4
            self.direction_vec = (1, 0)
        # If movement is 0 (no-op) or an invalid move (reversing), do nothing.

    def _check_collision(self, head):
        # Wall collision
        if not (0 <= head[0] < self.GRID_WIDTH and 0 <= head[1] < self.GRID_HEIGHT):
            return True
        # Self collision
        if head in list(self.snake_pos)[1:]:
            return True
        return False

    def _is_risky_move(self, pos):
        px, py = pos
        # Check adjacency to walls
        if px == 0 or px == self.GRID_WIDTH - 1 or py == 0 or py == self.GRID_HEIGHT - 1:
            return True
        # Check adjacency to self (excluding the segment that will move away)
        body_to_check = list(self.snake_pos)[:-1]
        for segment in body_to_check:
            if abs(px - segment[0]) + abs(py - segment[1]) == 1:
                return True
        return False

    def _place_food(self):
        all_cells = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        snake_cells = set(self.snake_pos)
        available_cells = list(all_cells - snake_cells)
        if not available_cells:
            # This should be extremely rare, but handle it to prevent crashes
            self.game_over = True
        else:
            # self.np_random.choice on a list of tuples returns a numpy array.
            # We convert it to a tuple to maintain type consistency with other coordinates.
            chosen_cell_array = self.np_random.choice(np.array(available_cells, dtype=object))
            self.food_pos = tuple(chosen_cell_array)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_pos),
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw food with glow
        food_rect = pygame.Rect(self.food_pos[0] * self.CELL_SIZE, 
                                self.food_pos[1] * self.CELL_SIZE, 
                                self.CELL_SIZE, self.CELL_SIZE)
        
        glow_radius = int(self.CELL_SIZE * 1.2)
        pygame.gfxdraw.filled_circle(self.screen, food_rect.centerx, food_rect.centery, glow_radius, self.COLOR_FOOD_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, food_rect.centerx, food_rect.centery, int(glow_radius*0.7), self.COLOR_FOOD_GLOW)
        
        pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect)

        # Draw snake
        # Head
        head = self.snake_pos[0]
        head_rect = pygame.Rect(head[0] * self.CELL_SIZE, head[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect)
        
        # Body
        for segment in list(self.snake_pos)[1:]:
            seg_rect = pygame.Rect(segment[0] * self.CELL_SIZE, segment[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, seg_rect.inflate(-4, -4))

    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        # Render reward popups
        new_popups = []
        for text, pos, lifetime, color in self.reward_popups:
            if lifetime > 0:
                alpha = int(255 * (lifetime / 30.0))
                popup_surf = self.font_small.render(text, True, color)
                popup_surf.set_alpha(alpha)
                
                pixel_pos = (pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2, 
                             pos[1] * self.CELL_SIZE - (30 - lifetime) // 2)
                popup_rect = popup_surf.get_rect(center=pixel_pos)
                
                self.screen.blit(popup_surf, popup_rect)
                new_popups.append((text, pos, lifetime - 1, color))
        self.reward_popups = new_popups

        # Render game over message
        if self.game_over:
            msg = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_REWARD_POS if self.score >= self.WIN_SCORE else self.COLOR_REWARD_NEG
            
            over_text = self.font_large.render(msg, True, color)
            over_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = over_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(over_text, over_rect)


    def _add_reward_popup(self, reward, pos):
        if abs(reward) < 0.01: return
        text = f"+{reward:.1f}" if reward > 0 else f"{reward:.1f}"
        color = self.COLOR_REWARD_POS if reward > 0 else self.COLOR_REWARD_NEG
        self.reward_popups.append((text, pos, 30, color)) # text, grid_pos, lifetime, color

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, you need to unset the headless environment variable
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        action = [0, 0, 0]  # Default action: no-op, no buttons
        
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
                elif event.key == pygame.K_q: # Quit on 'q'
                    running = False

        if not terminated:
            # Step the environment only when a key is pressed
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit human play speed

    env.close()