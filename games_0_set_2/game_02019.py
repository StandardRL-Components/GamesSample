
# Generated: 2025-08-28T03:25:36.336490
# Source Brief: brief_02019.md
# Brief Index: 2019

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space to jump. Time your jumps to land on the moving platforms and reach the green flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer where precise timing is key to navigating challenging platform layouts."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (48, 25, 52)
    COLOR_BG_BOTTOM = (24, 6, 48)
    COLOR_PLAYER = (255, 64, 64)
    COLOR_PLAYER_GLOW = (255, 100, 100)
    COLOR_PLATFORM = (240, 240, 240)
    COLOR_GOAL = (64, 255, 128)
    COLOR_TEXT = (255, 255, 255)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Physics
    GRAVITY = 0.3
    JUMP_STRENGTH = -8.0
    FPS = 60
    
    # Player
    PLAYER_SIZE = 20
    
    # Game
    TIME_LIMIT_SECONDS = 30
    TOTAL_STAGES = 3

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)
        
        # State variables that persist across episodes
        self.current_stage_num = 1
        self.last_episode_won = False
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.platforms = None
        self.platform_params = None
        self.goal_rect = None
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.game_won = False
        self.last_space_state = False
        self.on_platform = False
        self.landed_on_platform_ids = set()
        self.current_platform_id = -1
        
        # Initialize state
        # self.reset() # Called by validate_implementation

        # Validate implementation
        self.validate_implementation()
    
    def _generate_level(self):
        """Generates platforms for the current stage."""
        self.platforms = []
        self.platform_params = []

        # Starting platform is always static
        start_platform_width = 100
        start_platform = pygame.Rect(50, self.SCREEN_HEIGHT - 50, start_platform_width, 20)
        self.platforms.append(start_platform)
        self.platform_params.append({'type': 'static'})

        # Player starts on the first platform
        self.player_pos = [start_platform.centerx, start_platform.top - self.PLAYER_SIZE]

        # Stage-specific layouts
        base_amplitude = 100
        stage_amplitude = base_amplitude + (self.current_stage_num - 1) * 40.0

        if self.current_stage_num == 1:
            # Simple alternating platforms
            for i in range(1, 4):
                px = 150 + i * 120
                py = self.SCREEN_HEIGHT - 100 - i * 50
                rect = pygame.Rect(px - 40, py, 80, 20)
                self.platforms.append(rect)
                self.platform_params.append({
                    'type': 'sin',
                    'initial_x': rect.x,
                    'amplitude': self.np_random.uniform(20, 40),
                    'frequency': self.np_random.uniform(0.02, 0.04),
                    'phase': self.np_random.uniform(0, 2 * math.pi)
                })
        elif self.current_stage_num == 2:
            # Tighter jumps, more movement
            for i in range(1, 5):
                px = 120 + i * 110
                py = self.SCREEN_HEIGHT - 80 - i * 65
                rect = pygame.Rect(px - 35, py, 70, 20)
                self.platforms.append(rect)
                self.platform_params.append({
                    'type': 'sin',
                    'initial_x': rect.x,
                    'amplitude': self.np_random.uniform(40, stage_amplitude),
                    'frequency': self.np_random.uniform(0.03, 0.05),
                    'phase': self.np_random.uniform(0, 2 * math.pi)
                })
        elif self.current_stage_num == 3:
            # Challenging final stage
            for i in range(1, 6):
                px = 100 + i * 100
                py = self.SCREEN_HEIGHT - 60 - i * 60
                rect = pygame.Rect(px - 30, py, 60, 20)
                self.platforms.append(rect)
                self.platform_params.append({
                    'type': 'sin',
                    'initial_x': rect.x,
                    'amplitude': self.np_random.uniform(60, stage_amplitude),
                    'frequency': self.np_random.uniform(0.04, 0.06),
                    'phase': self.np_random.uniform(0, 2 * math.pi)
                })

        # Goal platform
        last_platform = self.platforms[-1]
        self.goal_rect = pygame.Rect(last_platform.centerx - 20, last_platform.top - 50, 40, 40)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Handle stage progression
        if self.last_episode_won:
            self.current_stage_num += 1
            if self.current_stage_num > self.TOTAL_STAGES:
                self.current_stage_num = 1 # Loop back
        self.last_episode_won = False # Reset for the new episode

        # Initialize game state
        self._generate_level()
        self.player_vel = [0, 0]
        
        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.game_over = False
        self.game_won = False
        
        self.last_space_state = False
        self.on_platform = True
        self.landed_on_platform_ids = {0} # Start on platform 0
        self.current_platform_id = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        space_held = action[1] == 1
        
        # --- Action Handling ---
        # Jump on space press (not hold)
        if space_held and not self.last_space_state and self.on_platform:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_platform = False
            self.current_platform_id = -1
            # Sound effect placeholder: player_jump.wav
        self.last_space_state = space_held
        
        # --- Physics and Game Logic Update ---
        self.steps += 1
        self.time_left -= 1

        # Update platform positions
        platform_deltas = [0] * len(self.platforms)
        for i, params in enumerate(self.platform_params):
            if params['type'] == 'sin':
                old_x = self.platforms[i].x
                new_x = params['initial_x'] + params['amplitude'] * math.sin(params['frequency'] * self.steps + params['phase'])
                self.platforms[i].x = new_x
                platform_deltas[i] = new_x - old_x

        # Update player
        if not self.on_platform:
            # Apply gravity
            self.player_vel[1] += self.GRAVITY
            # Continuous reward for airtime
            reward -= 0.01
        else:
            # Stick to platform
            self.player_vel[1] = 0
            if self.current_platform_id != -1:
                self.player_pos[0] += platform_deltas[self.current_platform_id]
                self.player_pos[1] = self.platforms[self.current_platform_id].top - self.PLAYER_SIZE
            # Continuous reward for being on a platform
            reward += 0.1

        # Update player position from velocity
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Collision detection
        if self.player_vel[1] > 0: # Only check for landing if falling
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat):
                    # Check if player's bottom is intersecting the top of the platform
                    if player_rect.bottom <= plat.top + self.player_vel[1] + 1:
                        self.player_pos[1] = plat.top - self.PLAYER_SIZE
                        self.player_vel[1] = 0
                        self.on_platform = True
                        self.current_platform_id = i
                        
                        if i not in self.landed_on_platform_ids:
                            self.landed_on_platform_ids.add(i)
                            reward += 10 # Event reward for new platform
                            # Sound effect placeholder: new_platform.wav
                        break
            else: # No collision
                self.on_platform = False

        # --- Termination Check ---
        terminated = False
        
        # 1. Reached goal
        if player_rect.colliderect(self.goal_rect):
            reward += 100
            terminated = True
            self.game_over = True
            self.game_won = True
            self.last_episode_won = True
            # Sound effect placeholder: win.wav

        # 2. Fell off screen
        if self.player_pos[1] > self.SCREEN_HEIGHT:
            reward -= 50
            terminated = True
            self.game_over = True
            # Sound effect placeholder: fall.wav

        # 3. Ran out of time
        if self.time_left <= 0:
            reward -= 50
            terminated = True
            self.game_over = True
            # Sound effect placeholder: time_out.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_background(self):
        """Draws a vertical gradient background."""
        for y in range(self.SCREEN_HEIGHT):
            # Interpolate color from top to bottom
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        """Renders all game elements."""
        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
            
        # Render goal
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_rect, border_radius=3)
        
        # Render player
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow effect
        glow_size = int(self.PLAYER_SIZE * 1.8)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size // 2, player_rect.centery - glow_size // 2))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        """Renders UI text overlays."""
        # Timer
        time_text = f"Time: {max(0, self.time_left // self.FPS):02d}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # Stage
        stage_text = f"Stage: {self.current_stage_num}/{self.TOTAL_STAGES}"
        stage_surf = self.font_ui.render(stage_text, True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (10, 10))

        # Game Over / Win Message
        if self.game_over:
            msg_text = "STAGE CLEAR!" if self.game_won else "TRY AGAIN"
            msg_color = self.COLOR_GOAL if self.game_won else self.COLOR_PLAYER
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_observation(self):
        """Gets the observation by rendering the game state."""
        self._render_background()
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage_num,
            "time_left": self.time_left // self.FPS,
        }
        
    def close(self):
        """Clean up resources."""
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        # We need to reset first to initialize everything for rendering
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Minimalist Platformer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Action mapping for human ---
        keys = pygame.key.get_pressed()
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        
        # The MultiDiscrete action is [movement, space, shift]
        # We only care about the space bar for this game
        action = [0, space_pressed, 0]

        # --- Step the environment ---
        obs, reward, term, trunc, info = env.step(action)
        
        if term:
            # Display final frame for a moment before reset
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000) # Wait 2 seconds
            obs, info = env.reset() # Reset for a new game

        # --- Rendering ---
        # The observation is the rendered frame, so we just display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Frame rate control ---
        clock.tick(GameEnv.FPS)

    env.close()