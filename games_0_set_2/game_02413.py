
# Generated: 2025-08-28T04:44:33.879860
# Source Brief: brief_02413.md
# Brief Index: 2413

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A side-view survival horror game where the player must collect keys in a dark
    room while being hunted by a procedurally animated shadow creature.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Collect all 5 keys and reach the white exit door. "
        "Avoid the shadow creature at all costs."
    )
    game_description = (
        "Escape a shadowy room by collecting keys while avoiding a lurking, procedurally-generated shadow creature. "
        "Tension and anticipation are your constant companions."
    )

    # Game configuration
    auto_advance = False

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 20
    GRID_W = SCREEN_WIDTH // GRID_SIZE
    GRID_H = SCREEN_HEIGHT // GRID_SIZE

    # Colors
    COLOR_BG = (20, 20, 25)
    COLOR_WALL = (40, 40, 50)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_KEY = (255, 255, 255)
    COLOR_EXIT_CLOSED = (100, 100, 120)
    COLOR_EXIT_OPEN = (255, 255, 255)
    COLOR_SHADOW = (0, 0, 0)
    COLOR_TEXT = (220, 220, 220)

    # Game rules
    TOTAL_KEYS = 5
    MAX_STEPS = 1000
    SHADOW_CHASE_RADIUS = 5  # in grid units

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = pygame.math.Vector2(0, 0)
        self.key_positions = []
        self.keys_collected = 0
        self.exit_pos = pygame.math.Vector2(0, 0)
        self.exit_open = False
        self.shadow_pos = pygame.math.Vector2(0, 0)
        self.shadow_spawn_pos = pygame.math.Vector2(0, 0)
        self.shadow_target_pos = pygame.math.Vector2(0, 0)
        self.shadow_state = "patrol"
        self.shadow_patrol_radius = 0
        self.shadow_speed = 0.0
        self.shadow_vertices = []
        self.shadow_noise_phase = 0.0
        self.dist_to_nearest_key = float('inf')

        # Validate implementation after setup
        # self.validate_implementation() # Optional: uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.exit_open = False
        self.keys_collected = 0
        
        # Player
        self.player_pos = pygame.math.Vector2(self.GRID_W // 2, self.GRID_H - 3)

        # Exit
        self.exit_pos = pygame.math.Vector2(self.GRID_W // 2, 1)

        # Keys
        self._spawn_keys()

        # Shadow
        self.shadow_spawn_pos = pygame.math.Vector2(self.GRID_W // 2, 5)
        self.shadow_pos = self.shadow_spawn_pos.copy()
        self.shadow_target_pos = self._get_new_patrol_target()
        self.shadow_state = "patrol"
        self.shadow_patrol_radius = 5.0
        self.shadow_speed = 0.5  # tiles per step
        self.shadow_noise_phase = self.np_random.uniform(0, 2 * math.pi)
        self._update_shadow_vertices()

        # RL state
        self.dist_to_nearest_key = self._get_dist_to_nearest_key()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        # --- Update Game Logic ---
        self.steps += 1
        reward = -0.1  # Cost of living

        # Store pre-move state for reward calculation
        dist_before_move = self._get_dist_to_nearest_key()

        # 1. Update Player
        self._update_player(movement)

        # 2. Update Shadow
        self._update_shadow()
        self._update_shadow_vertices()

        # 3. Check Interactions and Update State
        # Key collection
        collected_key = self._check_key_collection()
        if collected_key:
            reward += 10.0
            self.score += 10
            self.keys_collected += 1
            if self.keys_collected == self.TOTAL_KEYS:
                self.exit_open = True
                # Sound: key_all_collected.wav

        # Update distance-based reward
        dist_after_move = self._get_dist_to_nearest_key()
        if dist_after_move < dist_before_move:
            reward += 1.0
        self.dist_to_nearest_key = dist_after_move

        # 4. Check Termination Conditions
        terminated = False
        # Collision with shadow (lose)
        if self.player_pos.distance_to(self.shadow_pos) < 1.0:
            self.game_over = True
            terminated = True
            reward -= 100.0
            self.score -= 100
            # Sound: player_caught.wav

        # Reached exit (win)
        if self.exit_open and self.player_pos == self.exit_pos:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100.0
            self.score += 100
            # Sound: win_level.wav
        
        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    # --- Helper Methods for Logic ---

    def _update_player(self, movement):
        """Updates player position based on action, handling wall collisions."""
        target_pos = self.player_pos.copy()
        if movement == 1:  # Up
            target_pos.y -= 1
        elif movement == 2:  # Down
            target_pos.y += 1
        elif movement == 3:  # Left
            target_pos.x -= 1
        elif movement == 4:  # Right
            target_pos.x += 1
        
        # Wall collision check (1 tile padding)
        if 1 <= target_pos.x < self.GRID_W - 1 and 1 <= target_pos.y < self.GRID_H - 1:
            self.player_pos = target_pos

    def _update_shadow(self):
        """Updates shadow FSM, difficulty, and position."""
        # Difficulty scaling
        if self.steps > 0 and self.steps % 100 == 0:
            self.shadow_speed += 0.05
        if self.steps > 0 and self.steps % 200 == 0:
            self.shadow_patrol_radius += 1.0
        
        # FSM: Patrol vs Chase
        if self.player_pos.distance_to(self.shadow_pos) < self.SHADOW_CHASE_RADIUS:
            self.shadow_state = "chase"
            self.shadow_target_pos = self.player_pos
        else:
            self.shadow_state = "patrol"
            if self.shadow_pos.distance_to(self.shadow_target_pos) < 1.0:
                self.shadow_target_pos = self._get_new_patrol_target()

        # Movement
        if self.shadow_pos != self.shadow_target_pos:
            direction = (self.shadow_target_pos - self.shadow_pos).normalize()
            self.shadow_pos += direction * self.shadow_speed
            
        # Clamp to bounds to prevent escaping
        self.shadow_pos.x = max(1, min(self.GRID_W - 2, self.shadow_pos.x))
        self.shadow_pos.y = max(1, min(self.GRID_H - 2, self.shadow_pos.y))

    def _get_new_patrol_target(self):
        """Generates a new random target for the shadow in patrol mode."""
        while True:
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(0, self.shadow_patrol_radius)
            offset = pygame.math.Vector2(math.cos(angle) * radius, math.sin(angle) * radius)
            target = self.shadow_spawn_pos + offset
            # Ensure target is within bounds
            if 1 <= target.x < self.GRID_W - 1 and 1 <= target.y < self.GRID_H - 1:
                return target

    def _check_key_collection(self):
        """Checks if player is on a key and removes it if so."""
        for key_pos in self.key_positions:
            if self.player_pos == key_pos:
                self.key_positions.remove(key_pos)
                # Sound: key_collect.wav
                return True
        return False

    def _spawn_keys(self):
        """Spawns keys in valid, non-overlapping locations."""
        self.key_positions = []
        occupied_tiles = {tuple(self.player_pos), tuple(self.exit_pos)}
        
        for _ in range(self.TOTAL_KEYS):
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.integers(2, self.GRID_W - 2),
                    self.np_random.integers(2, self.GRID_H - 2),
                )
                if tuple(pos) not in occupied_tiles:
                    self.key_positions.append(pos)
                    occupied_tiles.add(tuple(pos))
                    break

    def _get_dist_to_nearest_key(self):
        """Calculates Euclidean distance to the nearest uncollected key."""
        if not self.key_positions:
            return self.player_pos.distance_to(self.exit_pos)
        
        min_dist = float('inf')
        for key_pos in self.key_positions:
            dist = self.player_pos.distance_to(key_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # --- Rendering Methods ---

    def _get_observation(self):
        """Renders the game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all game world elements."""
        # Draw room walls
        wall_rect = pygame.Rect(self.GRID_SIZE, self.GRID_SIZE, 
                                self.SCREEN_WIDTH - 2 * self.GRID_SIZE, 
                                self.SCREEN_HEIGHT - 2 * self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_WALL, wall_rect)

        # Draw exit
        exit_color = self.COLOR_EXIT_OPEN if self.exit_open else self.COLOR_EXIT_CLOSED
        exit_rect = pygame.Rect(self.exit_pos.x * self.GRID_SIZE, self.exit_pos.y * self.GRID_SIZE, 
                                self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, exit_color, exit_rect)

        # Draw keys
        for key_pos in self.key_positions:
            cx = key_pos.x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = key_pos.y * self.GRID_SIZE + self.GRID_SIZE // 2
            s = self.GRID_SIZE // 3
            points = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_KEY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_KEY)

        # Draw shadow
        if self.shadow_vertices:
            pygame.gfxdraw.aapolygon(self.screen, self.shadow_vertices, self.COLOR_SHADOW)
            pygame.gfxdraw.filled_polygon(self.screen, self.shadow_vertices, self.COLOR_SHADOW)

        # Draw player
        player_rect = pygame.Rect(self.player_pos.x * self.GRID_SIZE, self.player_pos.y * self.GRID_SIZE,
                                  self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _update_shadow_vertices(self):
        """Generates the animated blob shape for the shadow."""
        self.shadow_vertices = []
        num_vertices = 12
        center_x = self.shadow_pos.x * self.GRID_SIZE + self.GRID_SIZE / 2
        center_y = self.shadow_pos.y * self.GRID_SIZE + self.GRID_SIZE / 2
        base_radius = self.GRID_SIZE * 0.8
        
        self.shadow_noise_phase += 0.15
        
        for i in range(num_vertices):
            angle = (i / num_vertices) * 2 * math.pi
            # Use sine waves with different frequencies for a blobby, organic look
            noise = 0.3 * math.sin(self.shadow_noise_phase + angle * 3) + \
                    0.2 * math.cos(self.shadow_noise_phase * 0.5 + angle * 5)
            radius = base_radius * (1 + noise)
            
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            self.shadow_vertices.append((int(x), int(y)))

    def _render_ui(self):
        """Renders UI text like score and game over messages."""
        # Key counter
        key_text = f"Keys: {self.keys_collected} / {self.TOTAL_KEYS}"
        text_surface = self.font_small.render(key_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (15, 15))

        # Game Over message
        if self.game_over:
            message = "ESCAPED!" if self.win else "CAUGHT!"
            color = (180, 255, 180) if self.win else (255, 100, 100)
            msg_surface = self.font_large.render(message, True, color)
            msg_rect = msg_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surface, msg_rect)

    # --- Gymnasium Interface ---
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "keys_collected": self.keys_collected,
            "player_pos": tuple(self.player_pos),
            "shadow_pos": (self.shadow_pos.x, self.shadow_pos.y),
            "dist_to_nearest_key": self.dist_to_nearest_key
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
    
    # --- Validation ---

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert 'score' in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert 'steps' in info and info['steps'] == 1
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # Manual play loop
    obs, info = env.reset()
    done = False
    
    # For rendering to screen
    pygame.display.set_caption(GameEnv.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = [0, 0, 0] # No-op
    
    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
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
                    action = [0, 0, 0]
                    continue
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Reset action to no-op after one step for turn-based control
        action[0] = 0

        # --- Render ---
        # The observation is already the rendered image
        # We just need to get it from the env and display it
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if done:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            done = False

    env.close()