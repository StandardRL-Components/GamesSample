import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to direct the worm. Avoid walls and your own tail."

    # Must be a short, user-facing description of the game:
    game_description = "A classic arcade game. Guide the growing worm to eat food and score points. The game ends if you hit a wall or the worm's own body."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_WORM_BODY = (50, 205, 50)  # LimeGreen
    COLOR_WORM_HEAD = (124, 252, 0)  # LawnGreen
    COLOR_FOOD = (255, 69, 0)  # OrangeRed
    COLOR_TEXT = (255, 255, 255)
    COLOR_OVERLAY = (0, 0, 0, 180) # Semi-transparent black for game over

    MAX_SCORE = 100
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.worm_body = []
        self.direction = (0, 0)
        self.food_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.terminated = False
        self.dist_to_food = 0
        
        # Call reset to properly initialize the state
        # A default np_random is created here, which is fine.
        # It will be re-seeded if reset(seed=...) is called later.
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.terminated = False
        
        # Initialize worm
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.worm_body = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ]
        self.direction = (1, 0)  # Start moving right

        # Place food
        self._place_food()

        # Calculate initial distance to food for reward
        head_pos = self.worm_body[0]
        self.dist_to_food = abs(head_pos[0] - self.food_pos[0]) + abs(head_pos[1] - self.food_pos[1])
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Determine new direction, preventing reversal
        dx, dy = self.direction
        if movement == 1 and self.direction != (0, 1):    # Up
            dx, dy = 0, -1
        elif movement == 2 and self.direction != (0, -1):  # Down
            dx, dy = 0, 1
        elif movement == 3 and self.direction != (1, 0):   # Left
            dx, dy = -1, 0
        elif movement == 4 and self.direction != (-1, 0):  # Right
            dx, dy = 1, 0
        
        self.direction = (dx, dy)
        
        # Update game logic
        self.steps += 1
        
        # Calculate new head position
        head_x, head_y = self.worm_body[0]
        new_head = (head_x + dx, head_y + dy)
        
        # Check for termination conditions
        # 1. Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            self.terminated = True
            reward = 0
            return self._get_observation(), reward, self.terminated, False, self._get_info()

        # 2. Self-collision (check against all but the last segment, which will move)
        if new_head in self.worm_body[:-1]:
            self.terminated = True
            reward = 0
            return self._get_observation(), reward, self.terminated, False, self._get_info()

        # Move worm
        self.worm_body.insert(0, new_head)
        
        # Default reward is for movement relative to food
        new_dist_to_food = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
        if new_dist_to_food < self.dist_to_food:
            reward = 1.0  # Moved closer
        else:
            reward = -1.0 # Moved away or same
        self.dist_to_food = new_dist_to_food

        # Check for food consumption
        if new_head == self.food_pos:
            # Snake grows, so we don't pop the tail
            self.score += 10
            reward = 10.0
            if self.score >= self.MAX_SCORE:
                self.terminated = True
                reward = 100.0 # Win bonus
            else:
                self._place_food()
                # Recalculate dist_to_food for the new food position
                self.dist_to_food = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
        else:
            # Snake moves, so we pop the tail
            self.worm_body.pop()
        
        # 3. Max steps termination
        if self.steps >= self.MAX_STEPS:
            self.terminated = True
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _place_food(self):
        """Finds a random empty cell and places food there."""
        all_coords = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        worm_coords = set(self.worm_body)
        empty_coords = list(all_coords - worm_coords)
        if not empty_coords:
            # No space left, terminate game (unlikely but possible)
            self.terminated = True
            self.food_pos = (-1, -1)
        else:
            # np_random.choice on a list of tuples returns a numpy array.
            # This causes `new_head == self.food_pos` to fail with a ValueError.
            # To fix this, we select an index and get the tuple from the list,
            # ensuring self.food_pos remains a tuple.
            choice_index = self.np_random.integers(len(empty_coords))
            self.food_pos = empty_coords[choice_index]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw food
        if self.food_pos != (-1, -1):
            food_x = int(self.food_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
            food_y = int(self.food_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            radius = int(self.CELL_SIZE / 2 * 0.8)
            pygame.gfxdraw.filled_circle(self.screen, food_x, food_y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, food_x, food_y, radius, self.COLOR_FOOD)

        # Draw worm
        if not self.worm_body:
            return
            
        # Draw body segments
        for i, segment in enumerate(self.worm_body[1:]):
            x, y = segment
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WORM_BODY, rect, border_radius=3)
        
        # Draw head
        head_x, head_y = self.worm_body[0]
        head_rect = pygame.Rect(head_x * self.CELL_SIZE, head_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_WORM_HEAD, head_rect, border_radius=4)
        
        # Draw eyes on head
        eye_size = 3
        dx, dy = self.direction
        if dx == 1: # Right
            eye1_pos = (head_rect.centerx + 3, head_rect.centery - 5)
            eye2_pos = (head_rect.centerx + 3, head_rect.centery + 5)
        elif dx == -1: # Left
            eye1_pos = (head_rect.centerx - 3, head_rect.centery - 5)
            eye2_pos = (head_rect.centerx - 3, head_rect.centery + 5)
        elif dy == 1: # Down
            eye1_pos = (head_rect.centerx - 5, head_rect.centery + 3)
            eye2_pos = (head_rect.centerx + 5, head_rect.centery + 3)
        else: # Up or No-op
            eye1_pos = (head_rect.centerx - 5, head_rect.centery - 3)
            eye2_pos = (head_rect.centerx + 5, head_rect.centery - 3)
        pygame.draw.circle(self.screen, (0,0,0), eye1_pos, eye_size)
        pygame.draw.circle(self.screen, (0,0,0), eye2_pos, eye_size)


    def _render_ui(self):
        # Display score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Display game over/win message
        if self.terminated:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.MAX_SCORE:
                end_text = "YOU WIN!"
            else:
                end_text = "GAME OVER"
            
            text_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "worm_length": len(self.worm_body)
        }

    def close(self):
        pygame.quit()
        

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to action space
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # For human play, we need a display
    # We must unset the dummy video driver to see the window
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    # We control the game loop for human play
    # This is different from the RL agent loop
    current_movement = 0 # No-op
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    current_movement = key_to_action[event.key]
                if event.key == pygame.K_r and done:
                    # Reset game on 'R' key press if done
                    obs, info = env.reset()
                    done = False
                    current_movement = 0

        if not done:
            # Create the full action tuple
            action = (current_movement, 0, 0)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Reset movement so it doesn't repeat without new key press
            current_movement = 0
            
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # The game is not auto-advancing, so we wait for input.
        # But for human play, a small delay makes it playable.
        clock.tick(10) # 10 FPS feels right for classic snake

    env.close()