import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:10:45.080800
# Source Brief: brief_00932.md
# Brief Index: 932
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, color, life, dx, dy, radius, gravity=0.1):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.gravity = gravity

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.dy += self.gravity
        self.life -= 1

    def draw(self, surface, camera_y):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = self.color + (alpha,)
            pos = (int(self.x), int(self.y - camera_y))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.radius), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Jump from one oscillating platform to the next, climbing higher while avoiding the fall. "
        "Land on new platforms for a chance to gain temporary flight."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move left and right. "
        "Press space to jump, or to boost while flying."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 80)
    COLOR_PLATFORM = (180, 180, 190)
    COLOR_PLATFORM_MARKER = (60, 80, 120)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_FLY = (255, 255, 0)
    COLOR_PLAYER_GLOW = (150, 200, 255)
    COLOR_PLAYER_FLY_GLOW = (255, 255, 150)
    COLOR_TEXT = (255, 255, 255)
    
    # Player Physics
    PLAYER_SIZE = 12
    PLAYER_ACCEL = 1.0
    PLAYER_FRICTION = 0.85
    GRAVITY = 0.8
    JUMP_STRENGTH = -15
    MAX_HORIZ_SPEED = 8
    
    # Flight Mechanics
    FLIGHT_DURATION = 150 # 5 seconds at 30 FPS
    FLIGHT_CHANCE = 0.15 # 15% chance to get flight mode on a new platform
    FLIGHT_GRAVITY = 0.1
    FLIGHT_BOOST = -1.0

    # Platform Mechanics
    NUM_PLATFORMS = 30
    PLATFORM_HEIGHT = 15
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_grounded = False
        self.last_jump_on_platform = -1
        
        self.is_flying = False
        self.flight_timer = 0
        
        self.platforms = []
        self.current_platform_index = -1
        
        self.camera_y = 0.0
        self.particles = []
        
        self.last_space_held = False

        # self.validate_implementation() # Commented out for submission
        # self.reset() # Called by the wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player_pos = np.array([start_platform['rect'].centerx, float(start_platform['rect'].top - self.PLAYER_SIZE)])
        self.player_vel = np.array([0.0, 0.0])
        
        self.is_grounded = True
        self.current_platform_index = 0
        self.last_jump_on_platform = 0
        
        self.is_flying = False
        self.flight_timer = 0
        
        self.camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.7
        self.particles = []
        self.last_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # 1. Handle player input and movement
        self._handle_input(movement, space_held)

        # 2. Update player physics
        self._update_player_physics()
        
        # 3. Update game world
        self._update_platforms()
        self._update_particles()
        
        # 4. Collision detection and state update
        was_grounded = self.is_grounded
        self.is_grounded = False
        
        newly_landed_platform_index = self._check_collisions()
        
        if self.is_grounded:
            reward += 0.01 # Small reward for being on a platform
            if not was_grounded: # Just landed
                # Sound effect placeholder: # sfx_land.play()
                self._create_landing_particles(self.player_pos[0], self.player_pos[1] + self.PLAYER_SIZE)
                
            if newly_landed_platform_index > self.current_platform_index:
                reward += 1.0 + (newly_landed_platform_index - self.current_platform_index) # Bonus for skipping
                self.score += 10 * (newly_landed_platform_index - self.current_platform_index)
                self.current_platform_index = newly_landed_platform_index
                
                # Chance to activate flight mode
                if not self.is_flying and self.np_random.random() < self.FLIGHT_CHANCE:
                    self.is_flying = True
                    self.flight_timer = self.FLIGHT_DURATION
                    # Sound effect placeholder: # sfx_powerup.play()

        elif was_grounded: # Just fell off a platform
             reward -= 0.5

        # 5. Update camera
        self._update_camera()
        
        # 6. Check for termination conditions
        self.steps += 1
        terminated = False
        
        if self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT + 50:
            terminated = True
            reward -= 10.0 # Large penalty for falling off screen
        
        if self.current_platform_index >= self.NUM_PLATFORMS - 1:
            terminated = True
            reward += 100.0
            self.score += 1000
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.last_space_held = space_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3: # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player_vel[0] += self.PLAYER_ACCEL
            
        # Jumping
        jump_pressed = space_held and not self.last_space_held
        if jump_pressed and self.is_grounded:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_grounded = False
            self.last_jump_on_platform = self.current_platform_index
            # Sound effect placeholder: # sfx_jump.play()
            
        # Flying boost
        if self.is_flying and space_held:
            self.player_vel[1] += self.FLIGHT_BOOST
            # Emit boost particles
            if self.steps % 2 == 0:
                self.particles.append(Particle(
                    self.player_pos[0], self.player_pos[1] + self.PLAYER_SIZE, self.COLOR_PLAYER_FLY,
                    life=20, dx=self.np_random.uniform(-1, 1), dy=self.np_random.uniform(1, 3),
                    radius=self.np_random.uniform(2, 4), gravity=0.05
                ))

    def _update_player_physics(self):
        # Apply friction/damping
        self.player_vel[0] *= self.PLAYER_FRICTION
        if abs(self.player_vel[0]) < 0.1:
            self.player_vel[0] = 0
            
        # Clamp horizontal speed
        self.player_vel[0] = np.clip(self.player_vel[0], -self.MAX_HORIZ_SPEED, self.MAX_HORIZ_SPEED)
        
        # Apply gravity
        if self.is_flying:
            self.player_vel[1] += self.FLIGHT_GRAVITY
            self.flight_timer -= 1
            if self.flight_timer <= 0:
                self.is_flying = False
                # Sound effect placeholder: # sfx_powerdown.play()
        elif not self.is_grounded:
            self.player_vel[1] += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel
        
        # Screen bounds (horizontal)
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        if self.player_pos[0] == self.PLAYER_SIZE or self.player_pos[0] == self.SCREEN_WIDTH - self.PLAYER_SIZE:
            self.player_vel[0] = 0

    def _generate_platforms(self):
        self.platforms = []
        x_pos = self.SCREEN_WIDTH / 2
        y_pos = self.SCREEN_HEIGHT - 50
        
        for i in range(self.NUM_PLATFORMS):
            width = self.np_random.uniform(120, 180) - i * 2
            width = max(width, 60)
            
            if i > 0:
                prev_platform = self.platforms[i-1]
                dx = self.np_random.uniform(90, 150) + (i // 3) * 5
                x_pos = prev_platform['rect'].centerx + self.np_random.choice([-1, 1]) * dx
                
                # Clamp x_pos to be within screen bounds
                x_pos = np.clip(x_pos, width / 2 + 10, self.SCREEN_WIDTH - width / 2 - 10)
                
                dy = self.np_random.uniform(50, 120)
                y_pos -= dy

            amplitude = self.np_random.uniform(10, 30) + i * 0.5
            frequency = self.np_random.uniform(0.02, 0.05)
            phase = self.np_random.uniform(0, 2 * math.pi)
            
            rect = pygame.Rect(x_pos - width / 2, y_pos, width, self.PLATFORM_HEIGHT)
            
            self.platforms.append({
                'rect': rect,
                'y_center': y_pos,
                'amplitude': amplitude,
                'frequency': frequency,
                'phase': phase
            })

    def _update_platforms(self):
        for p in self.platforms:
            p['rect'].y = p['y_center'] + math.sin(self.steps * p['frequency'] + p['phase']) * p['amplitude']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE / 2,
            self.player_pos[1] - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        player_bottom = self.player_pos[1] + self.PLAYER_SIZE
        newly_landed_platform_index = -1

        # Check only platforms near the one we last jumped from
        start_check = max(0, self.last_jump_on_platform)
        end_check = min(len(self.platforms), start_check + 5)
        
        for i in range(start_check, end_check):
            p = self.platforms[i]
            platform_rect = p['rect']
            
            # Conditions for landing:
            # 1. Player is moving down
            # 2. Player's feet were above the platform last frame
            # 3. Player's feet are now on or below the platform
            # 4. Player is horizontally aligned with the platform
            if (self.player_vel[1] > 0 and
                player_bottom - self.player_vel[1] <= platform_rect.top and
                player_bottom >= platform_rect.top and
                player_rect.right > platform_rect.left and
                player_rect.left < platform_rect.right):
                
                self.is_grounded = True
                self.player_vel[1] = 0
                self.player_pos[1] = platform_rect.top - self.PLAYER_SIZE
                newly_landed_platform_index = i
                break
        
        return newly_landed_platform_index

    def _update_camera(self):
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.66
        # Smooth camera movement using linear interpolation
        self.camera_y += (target_cam_y - self.camera_y) * 0.1

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render platform oscillation markers
        for p in self.platforms:
            x = int(p['rect'].centerx)
            y_c = int(p['y_center'] - self.camera_y)
            amp = int(p['amplitude'])
            pygame.draw.line(self.screen, self.COLOR_PLATFORM_MARKER, (x, y_c - amp), (x, y_c + amp), 1)
        
        # Render platforms
        for p in self.platforms:
            cam_rect = p['rect'].copy()
            cam_rect.y -= self.camera_y
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, cam_rect, border_radius=3)
            
        # Render particles
        for particle in self.particles:
            particle.draw(self.screen, self.camera_y)
        
        # Render player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1] - self.camera_y)
        
        color = self.COLOR_PLAYER_FLY if self.is_flying else self.COLOR_PLAYER
        glow_color = self.COLOR_PLAYER_FLY_GLOW if self.is_flying else self.COLOR_PLAYER_GLOW
        
        # Glow effect
        for i in range(self.PLAYER_SIZE, 0, -2):
            alpha = 80 * (1 - i / self.PLAYER_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, i + 3, (*glow_color, alpha))

        # Player body
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.PLAYER_SIZE, color)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_SIZE, color)

    def _render_ui(self):
        # Platform counter
        platform_text = self.font.render(f"Platform: {self.current_platform_index + 1} / {self.NUM_PLATFORMS}", True, self.COLOR_TEXT)
        self.screen.blit(platform_text, (10, 10))
        
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Flight timer
        if self.is_flying:
            flight_seconds = self.flight_timer / self.FPS
            flight_text = self.small_font.render(f"Flight: {flight_seconds:.1f}s", True, self.COLOR_PLAYER_FLY)
            flight_rect = flight_text.get_rect(midbottom=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10))
            self.screen.blit(flight_text, flight_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "current_platform": self.current_platform_index,
            "is_flying": self.is_flying,
        }
        
    def _create_landing_particles(self, x, y):
        for _ in range(15):
            self.particles.append(Particle(
                x, y, self.COLOR_PLATFORM,
                life=self.np_random.integers(15, 30),
                dx=self.np_random.uniform(-2, 2),
                dy=self.np_random.uniform(-4, -1),
                radius=self.np_random.uniform(1, 4)
            ))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For human play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Platform Jumper")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        # The original code had UP/DOWN mapped, but they don't do anything in the logic.
        # We only need left/right.
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Episode finished in {info['steps']} steps.")
    print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    env.close()