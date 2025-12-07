
# Generated: 2025-08-27T21:01:59.682515
# Source Brief: brief_02654.md
# Brief Index: 2654

        
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

    # Short, user-facing control string:
    user_guide = (
        "Controls: Press Space to jump. Time your jumps to land on the platforms."
    )

    # Short, user-facing description of the game:
    game_description = (
        "A minimalist auto-runner. Jump through a procedurally generated world and try to reach the finish line."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (70, 130, 180)  # Steel Blue
    COLOR_PLAYER = (255, 255, 255)  # White
    COLOR_PLAYER_OUTLINE = (200, 200, 200)
    COLOR_PLATFORM = (128, 128, 128)  # Gray
    COLOR_PLATFORM_TOP = (160, 160, 160) # Lighter Gray
    COLOR_FINISH_POLE = (100, 100, 100)
    COLOR_FINISH_FLAG = (0, 200, 0) # Green
    COLOR_TEXT = (255, 255, 255)
    
    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game Physics & Rules
    GRAVITY = 0.5
    JUMP_STRENGTH = -10
    PLAYER_SPEED = 3.0
    PLAYER_SIZE = 20
    MAX_STEPS = 1000
    LEVEL_END_X = 5000
    
    # Procedural Generation
    INITIAL_PLATFORM_GAP = 80
    GAP_DIFFICULTY_INCREASE = 0.2
    DIFFICULTY_INTERVAL = 50

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
        self.font_ui = pygame.font.SysFont("Arial", 24)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.particles = None
        self.camera_x = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_space_held = None
        self.max_platform_gap = None
        
        self.reset()
        
        # Run validation check on initialization
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.camera_x = 0
        
        self.player_pos = np.array([100.0, 200.0])
        self.player_vel = np.array([self.PLAYER_SPEED, 0.0])
        self.on_ground = False
        
        self.particles = []
        self.platforms = []
        
        self.max_platform_gap = self.INITIAL_PLATFORM_GAP

        # Create starting platform
        start_platform = pygame.Rect(-self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 100, self.SCREEN_WIDTH * 2, 100)
        self.platforms.append(start_platform)
        
        # Generate initial platforms
        while self.platforms[-1].right < self.SCREEN_WIDTH * 1.5:
            self._generate_platform()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        # movement = action[0]  # 0-4: none/up/down/left/right (unused)
        space_held = action[1] == 1  # Boolean
        # shift_held = action[2] == 1  # Boolean (unused)
        
        reward = 0
        
        # --- Player Input ---
        is_jump_press = space_held and not self.last_space_held
        if is_jump_press:
            if self.on_ground:
                # sfx: Jump sound
                self.player_vel[1] = self.JUMP_STRENGTH
                self.on_ground = False
            else:
                # Penalize jumping in the air
                reward -= 0.2
        self.last_space_held = space_held

        # --- Physics and Movement ---
        old_x = self.player_pos[0]
        
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel
        
        # Reward for horizontal movement
        reward += (self.player_pos[0] - old_x) * 0.1

        # --- Collision Detection ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Assume not on ground until a collision is found
        landed_this_frame = False
        if self.player_vel[1] > 0: # Only check for landing if falling
            for platform in self.platforms:
                if player_rect.colliderect(platform):
                    # Check if the player's bottom was above the platform's top in the previous frame
                    prev_player_bottom = player_rect.bottom - self.player_vel[1]
                    if prev_player_bottom <= platform.top:
                        self.player_pos[1] = platform.top - self.PLAYER_SIZE
                        self.player_vel[1] = 0
                        landed_this_frame = True
                        # sfx: Landing thump
                        reward += 1
                        self._create_landing_particles(player_rect.midbottom)
                        break
        self.on_ground = landed_this_frame

        # --- Procedural Generation & Cleanup ---
        self._update_platforms()
        
        # --- Update Difficulty ---
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.max_platform_gap += self.GAP_DIFFICULTY_INCREASE

        # --- Update Particles ---
        self._update_particles()
        
        # --- Camera ---
        self.camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 4

        # --- Update Game State ---
        self.steps += 1
        terminated = False

        # Check for termination conditions
        if self.player_pos[1] > self.SCREEN_HEIGHT + self.PLAYER_SIZE:
            # sfx: Falling sound
            terminated = True
            reward = -10
        elif self.player_pos[0] >= self.LEVEL_END_X:
            # sfx: Victory fanfare
            terminated = True
            reward = 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        if terminated:
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_platform(self):
        last_platform = self.platforms[-1]
        gap = self.np_random.uniform(self.max_platform_gap * 0.5, self.max_platform_gap)
        
        width = self.np_random.integers(80, 250)
        height_variation = self.np_random.integers(-50, 50)
        
        new_x = last_platform.right + gap
        new_y = np.clip(last_platform.y + height_variation, 150, self.SCREEN_HEIGHT - 50)
        
        new_platform = pygame.Rect(new_x, new_y, width, self.SCREEN_HEIGHT - new_y)
        self.platforms.append(new_platform)

    def _update_platforms(self):
        # Generate new platforms if needed
        if self.platforms and self.platforms[-1].right - self.camera_x < self.SCREEN_WIDTH * 1.5:
            self._generate_platform()
            
        # Remove old platforms
        self.platforms = [p for p in self.platforms if p.right - self.camera_x > -100]

    def _create_landing_particles(self, pos):
        for _ in range(10):
            vel = np.array([self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)])
            life = self.np_random.integers(15, 30)
            size = self.np_random.integers(2, 5)
            self.particles.append({'pos': np.array(pos, dtype=float), 'vel': vel, 'life': life, 'size': size})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1 # Particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # --- Render all game elements ---
        self._render_background()
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a pleasant gradient
        for y in range(self.SCREEN_HEIGHT):
            mix_ratio = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[0] * mix_ratio,
                self.COLOR_BG_TOP[1] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[1] * mix_ratio,
                self.COLOR_BG_TOP[2] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[2] * mix_ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render Platforms
        for platform in self.platforms:
            on_screen_rect = platform.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, on_screen_rect)
            # Add a "top" surface for a slight 3D effect
            top_surface = pygame.Rect(on_screen_rect.left, on_screen_rect.top, on_screen_rect.width, 5)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, top_surface)

        # Render Finish Line
        finish_pole_x = self.LEVEL_END_X - self.camera_x
        if finish_pole_x < self.SCREEN_WIDTH + 50:
            finish_base_y = self.SCREEN_HEIGHT - 100 # Assuming a flat area near the end
            pygame.draw.line(self.screen, self.COLOR_FINISH_POLE, (finish_pole_x, finish_base_y), (finish_pole_x, finish_base_y - 80), 5)
            flag_points = [(finish_pole_x, finish_base_y - 80), (finish_pole_x + 40, finish_base_y - 60), (finish_pole_x, finish_base_y - 40)]
            pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FINISH_FLAG)

        # Render Particles
        for p in self.particles:
            on_screen_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = max(0, 255 * (p['life'] / 30.0))
            color = (*self.COLOR_PLAYER, alpha)
            
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (on_screen_pos[0] - p['size'], on_screen_pos[1] - p['size']))

        # Render Player
        player_screen_pos = (self.player_pos[0] - self.camera_x, self.player_pos[1])
        player_rect = pygame.Rect(int(player_screen_pos[0]), int(player_screen_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(4, 4))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_ui(self):
        # Render Timer
        time_elapsed = self.steps / 30.0 # Assuming 30 FPS
        timer_text = f"Time: {time_elapsed:.1f}s"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))
        
        # Render Progress
        progress = min(100, (self.player_pos[0] / self.LEVEL_END_X) * 100)
        progress_text = f"Progress: {progress:.0f}%"
        progress_surf = self.font_ui.render(progress_text, True, self.COLOR_TEXT)
        self.screen.blit(progress_surf, (self.SCREEN_WIDTH - progress_surf.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "progress_percent": (self.player_pos[0] / self.LEVEL_END_X) * 100
        }
        
    def close(self):
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
    # This part is for demonstration and will not be part of the final submission
    env = GameEnv(render_mode='rgb_array')
    env.validate_implementation()
    
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Minimalist Platformer")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Use a dictionary to track held keys for smooth input
    keys_held = {
        pygame.K_SPACE: False
    }

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # Map keyboard input to the MultiDiscrete action space
        if keys_held[pygame.K_SPACE]:
            action[1] = 1 # Space held

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to the display window ---
        # The observation is already a rendered frame
        # Need to transpose it back for pygame's display format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
        
        # Control the frame rate
        env.clock.tick(30)

    env.close()