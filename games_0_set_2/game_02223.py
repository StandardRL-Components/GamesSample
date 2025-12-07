
# Generated: 2025-08-28T04:07:57.435773
# Source Brief: brief_02223.md
# Brief Index: 2223

        
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

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press Space to dash right."
    )

    game_description = (
        "Navigate a procedurally generated pixel-art world. Jump and dash to reach the flag while avoiding obstacles and the void."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game constants
        self.FPS = 30
        self.GRAVITY = 0.5
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = -0.12
        self.PLAYER_JUMP_STRENGTH = -10
        self.PLAYER_DASH_STRENGTH = 15
        self.DASH_DURATION = 5 # frames
        self.DASH_COOLDOWN = 30 # frames
        self.MAX_FALL_SPEED = 15
        self.MAX_RUN_SPEED = 6
        self.LEVEL_END_X = 5000
        self.MAX_EPISODE_STEPS = 3000
        self.TIME_LIMIT_SECONDS = 180

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)
        self.COLOR_BG_BOTTOM = (240, 248, 255)
        self.COLOR_PLAYER = (255, 69, 0)
        self.COLOR_PLAYER_GLOW = (255, 165, 0, 50)
        self.COLOR_PLATFORM = (105, 105, 105)
        self.COLOR_OBSTACLE = (0, 0, 139)
        self.COLOR_FLAG = (0, 128, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = None
        self.coyote_time = None
        self.dash_timer = None
        self.dash_cooldown_timer = None
        self.platforms = None
        self.obstacles = None
        self.particles = None
        self.camera_x = None
        self.last_platform_x = None
        self.flag_rect = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.obstacle_spawn_prob = None
        self.platform_gap_avg = None

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS

        # Player state
        self.player_pos = pygame.math.Vector2(100, 200)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 20, 20)
        self.player_rect.center = self.player_pos
        self.on_ground = False
        self.coyote_time = 0
        self.dash_timer = 0
        self.dash_cooldown_timer = 0
        
        # World state
        self.camera_x = 0
        self.platforms = []
        self.obstacles = []
        self.particles = []
        self.last_platform_x = -50
        self.flag_rect = pygame.Rect(self.LEVEL_END_X, self.HEIGHT - 100, 20, 80)
        
        # Difficulty params
        self.obstacle_spawn_prob = 0.02
        self.platform_gap_avg = 50
        
        self._generate_initial_level()
        
        return self._get_observation(), self._get_info()

    def _generate_initial_level(self):
        # Create a starting platform
        start_platform = pygame.Rect(0, self.HEIGHT - 50, 200, 50)
        self.platforms.append(start_platform)
        self.last_platform_x = start_platform.right
        # Generate the world ahead
        while self.last_platform_x < self.WIDTH * 2:
            self._generate_world_chunk()

    def _generate_world_chunk(self):
        # Platform generation
        gap = self.np_random.uniform(self.platform_gap_avg * 0.8, self.platform_gap_avg * 1.2)
        width = self.np_random.uniform(80, 250)
        
        last_platform = self.platforms[-1]
        y_change = self.np_random.uniform(-80, 80)
        new_y = np.clip(last_platform.y + y_change, 150, self.HEIGHT - 30)

        new_platform_x = self.last_platform_x + gap
        new_platform = pygame.Rect(new_platform_x, new_y, width, self.HEIGHT - new_y)
        self.platforms.append(new_platform)
        self.last_platform_x = new_platform.right

        # Obstacle generation
        if self.np_random.random() < self.obstacle_spawn_prob and new_platform.width > 50:
            obstacle_width = 20
            obstacle_height = 20
            ox = new_platform.x + self.np_random.uniform(10, new_platform.width - obstacle_width - 10)
            oy = new_platform.y - obstacle_height
            # Add obstacle with properties: rect, type (0=static, 1=moving), phase, speed
            self.obstacles.append({
                "rect": pygame.Rect(ox, oy, obstacle_width, obstacle_height),
                "type": 1 if self.np_random.random() < 0.3 else 0, # 30% chance of being a moving obstacle
                "phase": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.np_random.uniform(0.5, 1.5),
                "initial_y": oy
            })
            
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # Time-based penalties and updates
        self.steps += 1
        self.time_remaining -= 1
        reward -= 1.0 / self.FPS * 0.01  # Small penalty for time passing

        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Player Physics
        landed_this_frame = self._update_player_state()
        if landed_this_frame:
            reward += 1.0
            self._create_particles(self.player_rect.midbottom, 15, self.COLOR_PLATFORM, count=10)

        # 3. Handle Collisions and generate rewards from them
        collision_reward, near_miss_reward = self._handle_collisions()
        reward += collision_reward + near_miss_reward

        # 4. Update World
        self._update_world_state()
        
        # 5. Reward for progress
        if self.player_vel.x > 0.1:
            reward += 0.01 * (self.player_pos.x / self.LEVEL_END_X)
        
        # 6. Check Termination
        terminated = False
        if self.player_rect.top > self.HEIGHT + 50:
            terminated = True
            reward = -100.0
        elif self.player_rect.colliderect(self.flag_rect):
            terminated = True
            reward = 100.0
        elif self.time_remaining <= 0 or self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            # No large penalty for timeout, just end the episode
        
        if terminated:
            self.game_over = True
            
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Horizontal Movement
        accel = pygame.math.Vector2(0, self.GRAVITY)
        if movement == 3: # Left
            accel.x = -self.PLAYER_ACCEL
        elif movement == 4: # Right
            accel.x = self.PLAYER_ACCEL
        
        # Friction
        accel.x += self.player_vel.x * self.PLAYER_FRICTION
        self.player_vel.x += accel.x
        if abs(self.player_vel.x) > self.MAX_RUN_SPEED and self.dash_timer <= 0:
            self.player_vel.x = math.copysign(self.MAX_RUN_SPEED, self.player_vel.x)
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0

        # Jumping
        if movement == 1 and (self.on_ground or self.coyote_time > 0):
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            self.coyote_time = 0
            self._create_particles(self.player_rect.midbottom, 10, self.COLOR_PLATFORM, count=5) # Jump dust
        
        # Dashing
        if space_held and self.dash_cooldown_timer <= 0:
            self.dash_timer = self.DASH_DURATION
            self.dash_cooldown_timer = self.DASH_COOLDOWN
            self.player_vel.x = self.PLAYER_DASH_STRENGTH
            self.player_vel.y = 0 # Momentary hover
            # Sound: Dash.wav

    def _update_player_state(self):
        was_on_ground = self.on_ground
        
        # Update dash timers
        if self.dash_timer > 0:
            self.dash_timer -= 1
            self._create_particles(self.player_rect.center, 5, self.COLOR_PLAYER_GLOW, count=3, life=10)
        if self.dash_cooldown_timer > 0:
            self.dash_cooldown_timer -= 1
        
        # Update velocity with gravity
        if not self.on_ground and self.dash_timer <= 0:
            self.player_vel.y += self.GRAVITY
        if self.player_vel.y > self.MAX_FALL_SPEED:
            self.player_vel.y = self.MAX_FALL_SPEED
            
        # Update position
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y

        # Prevent moving left off-screen
        if self.player_pos.x < self.player_rect.width / 2:
            self.player_pos.x = self.player_rect.width / 2
            self.player_vel.x = 0
            
        self.player_rect.center = self.player_pos
        
        # Coyote time
        if self.on_ground:
            self.coyote_time = 5 # 5 frames of grace period
        else:
            self.coyote_time -= 1

        return not was_on_ground and self.on_ground

    def _handle_collisions(self):
        collision_reward = 0
        near_miss_reward = 0
        self.on_ground = False

        # Platform collisions
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                # Player was above the platform in the previous frame
                if self.player_vel.y > 0 and self.player_pos.y - self.player_vel.y <= plat.top + self.player_rect.height / 2:
                    self.player_rect.bottom = plat.top
                    self.player_pos.y = self.player_rect.centery
                    self.player_vel.y = 0
                    self.on_ground = True
                # Hitting from the side
                elif self.player_rect.bottom > plat.top + 1:
                    if self.player_vel.x > 0: # Moving right
                        self.player_rect.right = plat.left
                    elif self.player_vel.x < 0: # Moving left
                        self.player_rect.left = plat.right
                    self.player_pos.x = self.player_rect.centerx
                    self.player_vel.x = 0
        
        # Obstacle collisions
        for obs in self.obstacles:
            # Near miss check
            near_miss_rect = obs["rect"].inflate(20, 20)
            if self.player_rect.colliderect(near_miss_rect) and not self.player_rect.colliderect(obs["rect"]):
                 near_miss_reward += 0.1 # Small reward per frame for being close

            if self.player_rect.colliderect(obs["rect"]):
                collision_reward = -5.0
                self.player_vel.x *= -0.5 # Knockback
                self.player_vel.y = -5 # Pop up
                self._create_particles(self.player_rect.center, 20, self.COLOR_OBSTACLE, count=20)
                # Sound: Hit.wav
                break # Only process one hit

        return collision_reward, near_miss_reward

    def _update_world_state(self):
        # Update camera
        self.camera_x = self.player_pos.x - self.WIDTH / 3
        
        # Update moving obstacles
        for obs in self.obstacles:
            if obs["type"] == 1: # Moving
                obs["rect"].y = obs["initial_y"] + math.sin(self.steps * 0.05 * obs["speed"] + obs["phase"]) * 20

        # Procedurally generate new chunks and remove old ones
        if self.player_pos.x > self.last_platform_x - self.WIDTH:
            self._generate_world_chunk()
            
        # Difficulty scaling
        if self.steps % 500 == 0 and self.steps > 0:
            self.obstacle_spawn_prob = min(0.1, self.obstacle_spawn_prob + 0.01)
            self.platform_gap_avg = min(100, self.platform_gap_avg + 1)
        
        # Clean up off-screen entities
        self.platforms = [p for p in self.platforms if p.right > self.camera_x]
        self.obstacles = [o for o in self.obstacles if o["rect"].right > self.camera_x]

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, pos, speed_mag, color, count=10, life=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_mag / 10
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({'pos': pygame.math.Vector2(pos), 'vel': vel, 'life': self.np_random.integers(life // 2, life), 'color': color})

    def _get_observation(self):
        self._render_background()
        self._render_world()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_world(self):
        # Draw platforms
        for plat in self.platforms:
            screen_rect = plat.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)

        # Draw obstacles
        for obs in self.obstacles:
            screen_rect = obs["rect"].move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
        
        # Draw flag
        screen_flag_rect = self.flag_rect.move(-self.camera_x, 0)
        if screen_flag_rect.colliderect(self.screen.get_rect()):
            pole_rect = pygame.Rect(screen_flag_rect.left, screen_flag_rect.top, 5, screen_flag_rect.height)
            flag_points = [
                (screen_flag_rect.left + 5, screen_flag_rect.top),
                (screen_flag_rect.left + 5, screen_flag_rect.top + 20),
                (screen_flag_rect.left + 25, screen_flag_rect.top + 10)
            ]
            pygame.draw.rect(self.screen, (150, 150, 150), pole_rect)
            pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Draw player
        screen_player_rect = self.player_rect.move(-self.camera_x, 0)
        
        # Squash and stretch effect
        draw_rect = screen_player_rect.copy()
        if not self.on_ground: # Stretch while falling
            draw_rect.height *= 1.2
            draw_rect.width *= 0.8
        if self.dash_timer > 0: # Stretch while dashing
            draw_rect.width *= 1.5
            draw_rect.height *= 0.7
        draw_rect.center = screen_player_rect.center

        # Glow effect
        glow_rect = draw_rect.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect())
        self.screen.blit(glow_surface, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, draw_rect)

    def _render_particles(self):
        for p in self.particles:
            screen_pos = p['pos'] - pygame.math.Vector2(self.camera_x, 0)
            size = max(1, p['life'] / 4)
            pygame.draw.circle(self.screen, p['color'], (int(screen_pos.x), int(screen_pos.y)), int(size))

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surface = font.render(text, True, color)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surface, pos)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        draw_text(score_text, self.font_small, self.COLOR_TEXT, (self.WIDTH - 150, 10))

        # Time
        time_seconds = self.time_remaining // self.FPS
        time_text = f"TIME: {time_seconds}"
        draw_text(time_text, self.font_small, self.COLOR_TEXT, (10, 10))
        
        if self.game_over:
            end_text = "LEVEL COMPLETE" if self.player_rect.colliderect(self.flag_rect) else "GAME OVER"
            draw_text(end_text, self.font_large, self.COLOR_TEXT, (self.WIDTH // 2 - 150, self.HEIGHT // 2 - 50))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining // self.FPS,
            "player_x": self.player_pos.x,
        }
        
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    
    # Set up window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    # Game loop
    while not terminated:
        # Map keyboard inputs to action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        # elif keys[pygame.K_DOWN]: # No down action
        #     movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        # Control frame rate
        env.clock.tick(env.FPS)

    env.close()
    pygame.quit()
    sys.exit()