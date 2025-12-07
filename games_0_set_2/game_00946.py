
# Generated: 2025-08-27T15:17:13.581908
# Source Brief: brief_00946.md
# Brief Index: 946

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    A Gymnasium environment for a fast-paced arcade puzzle game.
    The player navigates a 10x10 grid of disappearing tiles to reach a goal tile
    before the timer runs out. Each level increases the difficulty by making
    tiles disappear faster.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to jump between tiles. Reach the gold tile before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid of disappearing tiles to reach the goal tile within a time limit."
    )

    # Frames auto-advance for smooth animations and a real-time timer.
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    GRID_SIZE = 10
    TILE_SIZE = 36
    TILE_GAP = 4
    TOTAL_TILE_SIZE = TILE_SIZE + TILE_GAP
    
    GRID_WIDTH = GRID_SIZE * TOTAL_TILE_SIZE
    GRID_HEIGHT = GRID_SIZE * TOTAL_TILE_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_LINES = (40, 50, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_GOAL = (255, 215, 0)
    COLOR_TILE_SAFE = (40, 200, 120)
    COLOR_TILE_UNSTABLE = (255, 220, 0)
    COLOR_TILE_CRUMBLING = (255, 80, 50)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (10, 10, 15)

    # Game Mechanics
    NUM_LEVELS = 3
    LEVEL_TIME = 20.0
    INITIAL_DISAPPEAR_TIME = 2.0
    DIFFICULTY_INCREMENT = 0.25 # Time reduction per level
    RESPAWN_TIME = 3.0
    JUMP_DURATION_FRAMES = 8 # ~0.26s at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 72)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.player_pos = np.array([0, 0])
        self.goal_pos = np.array([0, 0])
        self.level_timer = 0.0
        self.disappear_duration = 0.0
        self.tile_timers = None
        self.tile_respawn_timers = None
        
        # Player animation state
        self.is_jumping = False
        self.jump_origin = np.array([0, 0])
        self.jump_target = np.array([0, 0])
        self.jump_frame = 0
        
        self.level_layouts = [
            {'start': (1, 5), 'goal': (8, 5)},
            {'start': (1, 1), 'goal': (8, 8)},
            {'start': (5, 8), 'goal': (5, 1)},
        ]
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def _setup_level(self):
        """Initializes the state for the current level."""
        layout = self.level_layouts[self.level - 1]
        self.player_pos = np.array(layout['start'], dtype=int)
        self.goal_pos = np.array(layout['goal'], dtype=int)
        
        self.level_timer = self.LEVEL_TIME
        self.disappear_duration = max(0.5, self.INITIAL_DISAPPEAR_TIME - (self.level - 1) * self.DIFFICULTY_INCREMENT)
        
        # Timers: positive value is countdown to disappear, negative is countdown to respawn
        self.tile_timers = np.full((self.GRID_SIZE, self.GRID_SIZE), self.disappear_duration + 1)
        
        self.is_jumping = False
        self.jump_frame = 0
        
        # Trigger the timer on the starting tile
        self.tile_timers[self.player_pos[0], self.player_pos[1]] = self.disappear_duration

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- UPDATE TIMERS ---
        self.steps += 1
        time_delta = 1.0 / self.FPS
        self.level_timer = max(0, self.level_timer - time_delta)

        # Update tile timers
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                # Active tiles counting down
                if 0 < self.tile_timers[x, y] <= self.disappear_duration:
                    self.tile_timers[x, y] -= time_delta
                # Respawning tiles
                elif self.tile_timers[x, y] < -time_delta:
                    self.tile_timers[x, y] += time_delta
                # Tile has just finished respawning
                elif -time_delta <= self.tile_timers[x, y] < 0:
                     self.tile_timers[x, y] = self.disappear_duration + 1 # Reset to safe state


        # --- HANDLE PLAYER MOVEMENT ---
        # Update ongoing jump
        if self.is_jumping:
            self.jump_frame += 1
            if self.jump_frame >= self.JUMP_DURATION_FRAMES:
                self.is_jumping = False
                self.player_pos = self.jump_target.copy()
                
                # Check for falling
                if not (0 <= self.player_pos[0] < self.GRID_SIZE and 0 <= self.player_pos[1] < self.GRID_SIZE) or \
                   self.tile_timers[self.player_pos[0], self.player_pos[1]] <= 0:
                    self.game_over = True
                    terminated = True
                    reward = -100 # Fall penalty
                    # sfx: player_fall
                else:
                    # sfx: player_land
                    # Landed safely, check tile state for reward
                    tile_time = self.tile_timers[self.player_pos[0], self.player_pos[1]]
                    if tile_time > self.disappear_duration * 0.5:
                        reward += 0.1 # Land on safe tile
                    elif tile_time > self.disappear_duration * 0.25:
                        reward += -0.5 # Land on unstable tile
                    else:
                        reward += -1.0 # Land on crumbling tile
                    
                    # Trigger the new tile's timer
                    self.tile_timers[self.player_pos[0], self.player_pos[1]] = self.disappear_duration

        # Process new action
        movement = action[0]
        if not self.is_jumping and movement != 0:
            direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
            move_vec = np.array(direction_map[movement])
            target_pos = self.player_pos + move_vec
            
            # Check if target is valid
            if 0 <= target_pos[0] < self.GRID_SIZE and 0 <= target_pos[1] < self.GRID_SIZE and \
               self.tile_timers[target_pos[0], target_pos[1]] > 0:
                self.is_jumping = True
                self.jump_frame = 0
                self.jump_origin = self.player_pos.copy()
                self.jump_target = target_pos.copy()
                # sfx: player_jump

        # --- CHECK GAME STATE ---
        # Check for level complete
        if np.array_equal(self.player_pos, self.goal_pos) and not self.is_jumping:
            reward += 5 # Level complete bonus
            self.score += int(self.level_timer * 10) # Time bonus
            
            if self.level < self.NUM_LEVELS:
                self.level += 1
                self._setup_level()
                # sfx: level_complete
            else: # Game won
                reward += 100 # Win game bonus
                self.game_over = True
                terminated = True
                # sfx: game_win

        # Check for timeout
        if self.level_timer <= 0:
            reward = -50 # Timeout penalty
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_X + grid_pos[0] * self.TOTAL_TILE_SIZE + self.TILE_SIZE / 2
        y = self.GRID_Y + grid_pos[1] * self.TOTAL_TILE_SIZE + self.TILE_SIZE / 2
        return np.array([x, y])

    def _render_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        shadow_surface = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        # Draw grid and tiles
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                px, py = self._grid_to_pixel((x, y))
                rect = pygame.Rect(px - self.TILE_SIZE / 2, py - self.TILE_SIZE / 2, self.TILE_SIZE, self.TILE_SIZE)
                
                timer_val = self.tile_timers[x, y]

                if timer_val > 0: # Tile is visible
                    if np.array_equal([x, y], self.goal_pos):
                        # Flashing goal tile
                        brightness = 0.75 + 0.25 * math.sin(self.steps * 0.3)
                        color = tuple(int(c * brightness) for c in self.COLOR_GOAL)
                    else:
                        # Regular tile color based on timer
                        if timer_val > self.disappear_duration:
                            color = self.COLOR_TILE_SAFE
                        elif timer_val > self.disappear_duration * 0.5:
                            color = self.COLOR_TILE_SAFE
                        elif timer_val > self.disappear_duration * 0.25:
                            color = self.COLOR_TILE_UNSTABLE
                        else: # Crumbling
                            color = self.COLOR_TILE_CRUMBLING
                    
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
                else: # Tile is gone
                    pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, border_radius=4)

        # Draw player
        if self.is_jumping:
            progress = self.jump_frame / self.JUMP_DURATION_FRAMES
            # Ease-out curve for smooth landing
            eased_progress = 1 - (1 - progress)**3
            start_pixel = self._grid_to_pixel(self.jump_origin)
            end_pixel = self._grid_to_pixel(self.jump_target)
            render_pos = start_pixel + (end_pixel - start_pixel) * eased_progress
            
            # Add a hop effect
            hop = math.sin(progress * math.pi) * 15
            render_pos[1] -= hop
            size_mod = 1 + math.sin(progress * math.pi) * 0.2
        else:
            render_pos = self._grid_to_pixel(self.player_pos)
            size_mod = 1.0

        player_radius = int(self.TILE_SIZE / 2.5 * size_mod)
        
        # Draw player glow
        for i in range(player_radius, 0, -3):
            alpha = 40 * (1 - i / player_radius)
            pygame.gfxdraw.filled_circle(
                self.screen, int(render_pos[0]), int(render_pos[1]), i + 3, (*self.COLOR_PLAYER, int(alpha))
            )
        
        # Draw player circle
        pygame.gfxdraw.filled_circle(
            self.screen, int(render_pos[0]), int(render_pos[1]), player_radius, self.COLOR_PLAYER
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(render_pos[0]), int(render_pos[1]), player_radius, self.COLOR_PLAYER
        )

        # Draw UI
        self._render_text(f"SCORE: {self.score}", (20, 15), self.font_ui)
        self._render_text(f"LEVEL: {self.level}/{self.NUM_LEVELS}", (self.SCREEN_WIDTH // 2 - 50, 15), self.font_ui)
        time_str = f"TIME: {self.level_timer:.1f}"
        self._render_text(time_str, (self.SCREEN_WIDTH - 150, 15), self.font_ui)

        # Draw Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            if np.array_equal(self.player_pos, self.goal_pos) and self.level == self.NUM_LEVELS:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            self._render_text(msg, (self.SCREEN_WIDTH // 2 - self.font_title.size(msg)[0] // 2, 
                                    self.SCREEN_HEIGHT // 2 - 50), self.font_title)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_pos": tuple(self.player_pos),
            "time_left": round(self.level_timer, 2)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to display the window
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # Monkey-patch the render method for human display
        GameEnv.metadata["render_modes"].append("human")
        def render_human(self):
            if not hasattr(self, 'display_screen'):
                pygame.display.init()
                self.display_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Tile Jumper")
            
            # Get the current frame as a surface and display it
            frame_surface = pygame.surfarray.make_surface(np.transpose(self._get_observation(), (1, 0, 2)))
            self.display_screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.FPS)
        GameEnv.render = render_human

    env = GameEnv(render_mode=render_mode)
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    # --- Manual Play Loop ---
    if render_mode == "human":
        action = np.array([0, 0, 0]) # Start with no-op
        while not terminated:
            # Get keyboard input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = np.array([movement, space_held, shift_held])
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("Game Over!")
                pygame.time.wait(2000)

    # --- Agent Random Play Loop ---
    else:
        total_reward = 0
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"Episode finished. Total reward: {total_reward}. Final score: {info['score']}")
                obs, info = env.reset()
                total_reward = 0

    env.close()