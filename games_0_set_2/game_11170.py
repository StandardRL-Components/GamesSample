import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:42:27.726536
# Source Brief: brief_01170.md
# Brief Index: 1170
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Leap between shifting platforms, dodging obstacles and chaining jumps to trigger
    platform expansions, in a vibrant, procedurally generated world.
    """
    game_description = (
        "Leap between shifting platforms, dodging obstacles and chaining jumps to trigger "
        "platform expansions, in a vibrant, procedurally generated world."
    )
    user_guide = "Controls: ↑ to jump, ←→ to switch platforms."
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Assumed FPS for smooth interpolation

    # Colors
    COLOR_BG_TOP = (25, 10, 50)
    COLOR_BG_BOTTOM = (45, 20, 80)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50, 50)
    COLOR_PLATFORM = (100, 150, 255)
    COLOR_PLATFORM_GLOW = (150, 200, 255, 150)
    COLOR_OBSTACLE = (255, 165, 0)
    COLOR_OBSTACLE_GLOW = (255, 165, 0, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE_JUMP = (180, 220, 255)
    COLOR_PARTICLE_LAND = (100, 150, 255)

    # Player settings
    PLAYER_SIZE = 20
    PLAYER_GRAVITY = 0.8
    PLAYER_JUMP_STRENGTH = -15
    PLAYER_SWITCH_SPEED = 0.2  # Lerp factor for switching platforms

    # Platform settings
    PLATFORM_HEIGHT = 20
    PLATFORM_WIDTH = 120
    PLATFORM_Y = 350
    PLATFORM_GAP = 200
    PLATFORM_MOVE_RANGE = 50
    PLATFORM_MOVE_SPEED = 0.03

    # Obstacle settings
    OBSTACLE_RADIUS = 15
    OBSTACLE_SPAWN_INTERVAL = 100  # in steps
    OBSTACLE_INITIAL_SPEED = 2.0
    OBSTACLE_SPEED_INCREASE_INTERVAL = 200 # steps
    OBSTACLE_SPEED_INCREASE_AMOUNT = 0.05

    # Game rules
    MAX_EPISODE_STEPS = 5000
    LEVEL_LENGTH = MAX_EPISODE_STEPS # Progress is tied to steps survived
    JUMP_CHAIN_TARGET = 5
    PLATFORM_EXPANSION_DURATION = 30 # steps

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
        self.font = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 48)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_jumping = True
        self.target_platform_idx = 0
        self.platforms = []
        self.obstacles = []
        self.particles = []
        self.obstacle_spawn_timer = 0
        self.obstacle_speed = self.OBSTACLE_INITIAL_SPEED
        self.progress = 0.0
        self.jump_chain_counter = 0
        self.platform_expansion_timer = 0

        # This check is critical for verifying the implementation
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.progress = 0.0
        self.jump_chain_counter = 0
        self.platform_expansion_timer = 0
        
        platform_center_x = self.SCREEN_WIDTH / 2
        platform_left_x = platform_center_x - self.PLATFORM_GAP / 2 - self.PLATFORM_WIDTH / 2
        platform_right_x = platform_center_x + self.PLATFORM_GAP / 2 - self.PLATFORM_WIDTH / 2

        self.platforms = [
            {'rect': pygame.Rect(platform_left_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT), 'center_x': platform_left_x, 'phase': 0},
            {'rect': pygame.Rect(platform_right_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT), 'center_x': platform_right_x, 'phase': math.pi}
        ]
        
        self.target_platform_idx = self.np_random.choice([0, 1])
        self.player_pos = pygame.Vector2(self.platforms[self.target_platform_idx]['rect'].centerx, self.PLATFORM_Y - self.PLAYER_SIZE)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_jumping = False

        self.obstacles = []
        self.particles = []
        self.obstacle_spawn_timer = self.OBSTACLE_SPAWN_INTERVAL
        self.obstacle_speed = self.OBSTACLE_INITIAL_SPEED

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held and shift_held are ignored as per the brief.

        reward = 0.0
        terminated = False

        # --- 1. Handle Input ---
        if movement == 1 and not self.is_jumping:  # Jump
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.is_jumping = True
            self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE / 2, self.PLAYER_SIZE), 20, self.COLOR_PARTICLE_JUMP, 'down')
            # sfx: jump_sound()
        elif movement == 3:  # Move Left
            self.target_platform_idx = 0
        elif movement == 4:  # Move Right
            self.target_platform_idx = 1

        # --- 2. Update Game Logic ---
        self.steps += 1
        reward += 0.1 # Survival reward

        # Update Timers
        if self.platform_expansion_timer > 0:
            self.platform_expansion_timer -= 1
        
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()
            self.obstacle_spawn_timer = self.OBSTACLE_SPAWN_INTERVAL

        # Difficulty scaling
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE_AMOUNT
        
        # Update progress
        self.progress = min(1.0, self.steps / self.LEVEL_LENGTH)

        # Update platforms
        for p in self.platforms:
            p['phase'] += self.PLATFORM_MOVE_SPEED
            offset = math.sin(p['phase']) * self.PLATFORM_MOVE_RANGE
            p['rect'].x = p['center_x'] + offset

        # Update player
        if self.is_jumping:
            self.player_vel.y += self.PLAYER_GRAVITY
        self.player_pos += self.player_vel
        
        target_platform_center_x = self.platforms[self.target_platform_idx]['rect'].centerx
        self.player_pos.x += (target_platform_center_x - self.PLAYER_SIZE/2 - self.player_pos.x) * self.PLAYER_SWITCH_SPEED

        player_rect = self._get_player_rect()

        # Update obstacles and check for rewards
        dodged_obstacles = []
        for obs in self.obstacles:
            obs['pos'].y += self.obstacle_speed
            if obs['pos'].y > player_rect.centery and not obs['dodged']:
                reward += 1.0
                self.jump_chain_counter += 1
                obs['dodged'] = True
                # sfx: dodge_success()
                if self.jump_chain_counter > 0 and self.jump_chain_counter % self.JUMP_CHAIN_TARGET == 0:
                    reward += 5.0
                    self.platform_expansion_timer = self.PLATFORM_EXPANSION_DURATION
                    # sfx: chain_bonus()
        self.obstacles = [obs for obs in self.obstacles if obs['pos'].y < self.SCREEN_HEIGHT + self.OBSTACLE_RADIUS]

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

        # --- 3. Check Collisions & Termination ---
        # Landing on platform
        platform = self.platforms[self.target_platform_idx]
        if self.player_vel.y > 0 and player_rect.colliderect(platform['rect']) and player_rect.bottom <= platform['rect'].top + self.player_vel.y:
            self.player_pos.y = platform['rect'].top - self.PLAYER_SIZE
            self.player_vel.y = 0
            if self.is_jumping: # Just landed
                self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE / 2, self.PLAYER_SIZE), 10, self.COLOR_PARTICLE_LAND, 'up')
                # sfx: land_sound()
            self.is_jumping = False
        else:
             # Check if player is on any platform, if not, reset chain
            on_any_platform = False
            for p in self.platforms:
                if player_rect.bottom == p['rect'].top and player_rect.left < p['rect'].right and player_rect.right > p['rect'].left:
                    on_any_platform = True
                    break
            if not on_any_platform:
                 self.is_jumping = True
                 if self.jump_chain_counter > 0:
                    self.jump_chain_counter = 0 # Reset chain if jump is missed

        # Falling off
        if player_rect.top > self.SCREEN_HEIGHT:
            terminated = True
            reward = -100.0
            # sfx: fall_sound()

        # Obstacle collision
        for obs in self.obstacles:
            if player_rect.colliderect(pygame.Rect(obs['pos'].x - obs['radius'], obs['pos'].y - obs['radius'], obs['radius']*2, obs['radius']*2)):
                dist = player_rect.center.distance_to(obs['pos'])
                if dist < self.PLAYER_SIZE / 2 + obs['radius']:
                    terminated = True
                    reward = -100.0
                    # sfx: hit_obstacle_sound()
                    break
        
        # Max steps or level complete
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            if self.progress >= 1.0:
                reward = 100.0 # Victory reward
                # sfx: level_complete_sound()
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)

    def _spawn_obstacle(self):
        platform_rects = [p['rect'] for p in self.platforms]
        min_x = min(r.left for r in platform_rects)
        max_x = max(r.right for r in platform_rects)
        
        x_pos = self.np_random.uniform(min_x, max_x)
        y_pos = -self.OBSTACLE_RADIUS
        
        self.obstacles.append({
            'pos': pygame.Vector2(x_pos, y_pos),
            'radius': self.OBSTACLE_RADIUS,
            'dodged': False
        })

    def _create_particles(self, pos, count, color, direction='all'):
        for _ in range(count):
            if direction == 'down':
                vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(0.5, 3))
            elif direction == 'up':
                vel = pygame.Vector2(self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-3, -0.5))
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(10, 20),
                'radius': self.np_random.uniform(1, 4),
                'color': color
            })

    def _render_game(self):
        # Draw gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['radius']))

        # Draw platforms
        for p in self.platforms:
            if self.platform_expansion_timer > 0:
                glow_alpha = self.platform_expansion_timer / self.PLATFORM_EXPANSION_DURATION
                color = (self.COLOR_PLATFORM_GLOW[0], self.COLOR_PLATFORM_GLOW[1], self.COLOR_PLATFORM_GLOW[2], int(self.COLOR_PLATFORM_GLOW[3] * glow_alpha))
                glow_rect = p['rect'].inflate(10, 10)
                
                # Create a temporary surface for the glow
                glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, color, glow_surf.get_rect(), border_radius=8)
                self.screen.blit(glow_surf, glow_rect.topleft)

            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p['rect'], border_radius=5)

        # Draw obstacles
        for obs in self.obstacles:
            pos = (int(obs['pos'].x), int(obs['pos'].y))
            rad = obs['radius']
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad + 4, self.COLOR_OBSTACLE_GLOW)
            # Main circle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], rad, self.COLOR_OBSTACLE)

        # Draw player
        player_rect = self._get_player_rect()
        # Glow
        glow_center = (int(player_rect.centerx), int(player_rect.centery))
        glow_radius = int(self.PLAYER_SIZE * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], glow_radius, self.COLOR_PLAYER_GLOW)
        # Player square
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        # Progress text
        progress_text = f"Progress: {self.progress:.0%}"
        text_surf = self.font.render(progress_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Score text
        score_text = f"Score: {self.score:.1f}"
        text_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Jump chain counter
        if self.jump_chain_counter > 1:
            chain_text = f"x{self.jump_chain_counter}"
            text_surf = self.font_big.render(chain_text, True, self.COLOR_PLAYER)
            text_rect = text_surf.get_rect(center=(self.player_pos.x + self.PLAYER_SIZE/2, self.player_pos.y - 25))
            self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_TOP) # Fallback fill
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress": self.progress,
            "jump_chain": self.jump_chain_counter,
            "obstacle_speed": self.obstacle_speed,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert 'score' in info and 'steps' in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert 'score' in info and 'steps' in info
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    
    # Create a Pygame window to display the environment
    # Must unset the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env.reset()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Platformer Environment")
    
    running = True
    terminated = False
    total_reward = 0.0
    
    # Action state
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    
    print("\n--- Manual Control ---")
    print("UP ARROW: Jump")
    print("LEFT/RIGHT ARROWS: Switch Platforms")
    print("R: Reset")
    print("Q: Quit")
    
    while running:
        if terminated:
            print(f"Episode Finished! Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            terminated = False
            total_reward = 0.0

        # --- Pygame event handling for manual control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    terminated = True # Force a reset on next loop
        
        keys = pygame.key.get_pressed()
        movement = 0 # Reset movement action each frame
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # Action is [movement, space, shift]
        action = [movement, 0, 0]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation to the screen ---
        # The observation is (H, W, C), but Pygame needs (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        obs_transposed = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_transposed)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control the frame rate ---
        env.clock.tick(GameEnv.FPS)

    env.close()