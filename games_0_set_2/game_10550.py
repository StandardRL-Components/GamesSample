import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a bouncing ball.
    The goal is to hit blocks to accumulate a target distance within a time limit.
    The ball can transform into a larger, more powerful state.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Control a bouncing ball to break blocks and achieve a target distance before time runs out."
    user_guide = "Use the arrow keys (↑↓←→) to apply force to the ball."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 50  # Steps per second
    MAX_STEPS = 30 * FPS  # 30 seconds episode
    WIN_DISTANCE = 1000

    # Colors
    COLOR_BG_START = (10, 20, 40)
    COLOR_BG_END = (30, 40, 70)
    COLOR_WALL = (200, 200, 220)
    COLOR_BLOCK = (220, 50, 50)
    COLOR_BALL_SMALL = (50, 220, 50)
    COLOR_BALL_LARGE = (255, 150, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (255, 80, 80)

    # Physics & Gameplay
    BALL_RADIUS_SMALL = 10
    BALL_RADIUS_LARGE = 16
    BALL_ACCELERATION = 0.25
    BALL_MAX_VEL = 6.0
    BALL_DRAG = 0.99
    BLOCK_SIZE = (40, 15)
    NUM_BLOCKS = 25
    BLOCKS_TO_TRANSFORM = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)

        # Pre-render background for performance
        self.background = self._create_gradient_background()

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.total_distance = 0.0
        self.game_over = False
        self.ball_pos = np.zeros(2, dtype=float)
        self.ball_vel = np.zeros(2, dtype=float)
        self.ball_is_large = False
        self.blocks_hit_since_transform = 0
        self.blocks = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.total_distance = 0.0
        self.game_over = False
        
        self.ball_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.ball_vel = np.array(
            [self.np_random.uniform(-2.5, 2.5), self.np_random.uniform(-2.5, 2.5)],
            dtype=float
        )
        
        self.ball_is_large = False
        self.blocks_hit_since_transform = 0
        self.blocks = self._generate_blocks()
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. UNPACK ACTION & INITIALIZE REWARD ---
        movement = action[0]
        reward = 0.0

        # --- 2. UPDATE GAME LOGIC ---
        self._update_ball(movement)
        collision_reward = self._handle_collisions()
        reward += collision_reward
        self._update_particles()

        # --- 3. HANDLE TRANSFORMATION ---
        if self.blocks_hit_since_transform >= self.BLOCKS_TO_TRANSFORM:
            if not self.ball_is_large:
                self.ball_is_large = True
                reward += 5.0  # Transformation reward
                self._spawn_particles(self.ball_pos, 30, self.COLOR_BALL_LARGE, 1.5)
            self.blocks_hit_since_transform = 0

        # --- 4. UPDATE STEP COUNTER & CHECK TERMINATION ---
        self.steps += 1
        terminated = (self.steps >= self.MAX_STEPS or
                      self.total_distance >= self.WIN_DISTANCE)
        truncated = False
        
        if terminated:
            self.game_over = True
            if self.total_distance >= self.WIN_DISTANCE:
                reward += 100.0 # Victory reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_ball(self, movement):
        # Apply acceleration from action
        if movement == 1:  # Up
            self.ball_vel[1] -= self.BALL_ACCELERATION
        elif movement == 2:  # Down
            self.ball_vel[1] += self.BALL_ACCELERATION
        elif movement == 3:  # Left
            self.ball_vel[0] -= self.BALL_ACCELERATION
        elif movement == 4:  # Right
            self.ball_vel[0] += self.BALL_ACCELERATION

        # Apply drag
        self.ball_vel *= self.BALL_DRAG

        # Clamp velocity
        speed = np.linalg.norm(self.ball_vel)
        if speed > self.BALL_MAX_VEL:
            self.ball_vel = self.ball_vel / speed * self.BALL_MAX_VEL

        # Update position
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        reward = 0.0
        radius = self.BALL_RADIUS_LARGE if self.ball_is_large else self.BALL_RADIUS_SMALL

        # Wall collisions
        if self.ball_pos[0] - radius < 0:
            self.ball_pos[0] = radius
            self.ball_vel[0] *= -1
        elif self.ball_pos[0] + radius > self.WIDTH:
            self.ball_pos[0] = self.WIDTH - radius
            self.ball_vel[0] *= -1
        if self.ball_pos[1] - radius < 0:
            self.ball_pos[1] = radius
            self.ball_vel[1] *= -1
        elif self.ball_pos[1] + radius > self.HEIGHT:
            self.ball_pos[1] = self.HEIGHT - radius
            self.ball_vel[1] *= -1

        # Block collisions
        ball_rect = pygame.Rect(self.ball_pos[0] - radius, self.ball_pos[1] - radius, radius * 2, radius * 2)
        
        for block in self.blocks[:]:
            if ball_rect.colliderect(block):
                self._spawn_particles(np.array(block.center), 15, self.COLOR_PARTICLE)
                self.blocks.remove(block)
                
                # Calculate rewards
                distance_gain = np.linalg.norm(self.ball_vel) * 2.5
                self.total_distance += distance_gain
                reward += distance_gain * 0.1  # Distance reward
                reward += 1.0  # Block destruction reward
                
                # Bounce logic
                dx = self.ball_pos[0] - block.centerx
                dy = self.ball_pos[1] - block.centery
                
                if abs(dx) > abs(dy): # Horizontal collision
                    self.ball_vel[0] *= -1
                    self.ball_pos[0] += np.sign(dx) * 2 # Push out
                else: # Vertical collision
                    self.ball_vel[1] *= -1
                    self.ball_pos[1] += np.sign(dy) * 2 # Push out
                
                self.blocks_hit_since_transform += 1
                break # Handle one collision per frame for simplicity

        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)

    def _get_observation(self):
        # 1. Clear screen with background
        self.screen.blit(self.background, (0, 0))

        # 2. Render game elements
        self._render_particles()
        self._render_blocks()
        self._render_ball()
        self._render_walls()

        # 3. Render UI overlay
        self._render_ui()

        # 4. Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.total_distance,
            "steps": self.steps,
            "ball_is_large": self.ball_is_large,
            "blocks_remaining": len(self.blocks),
        }
        
    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - ratio) + self.COLOR_BG_END[0] * ratio),
                int(self.COLOR_BG_START[1] * (1 - ratio) + self.COLOR_BG_END[1] * ratio),
                int(self.COLOR_BG_START[2] * (1 - ratio) + self.COLOR_BG_END[2] * ratio)
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_walls(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)

    def _render_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, block, border_radius=3)

    def _render_ball(self):
        pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        if self.ball_is_large:
            color = self.COLOR_BALL_LARGE
            radius = self.BALL_RADIUS_LARGE
        else:
            color = self.COLOR_BALL_SMALL
            radius = self.BALL_RADIUS_SMALL
        
        # Glow effect
        glow_radius = int(radius * 1.8)
        glow_alpha = 60
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main ball with anti-aliasing
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], pos, radius)

    def _render_ui(self):
        # Distance
        dist_text = self.font_large.render(f"DIST: {int(self.total_distance)}", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (15, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 15, 10))

        # Transformation meter
        if not self.ball_is_large:
            meter_width = 100
            meter_height = 10
            fill_width = int((self.blocks_hit_since_transform / self.BLOCKS_TO_TRANSFORM) * meter_width)
            meter_rect = pygame.Rect((self.WIDTH / 2 - meter_width / 2, 15), (meter_width, meter_height))
            fill_rect = pygame.Rect(meter_rect.topleft, (fill_width, meter_height))
            pygame.draw.rect(self.screen, (100,100,100), meter_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BALL_LARGE, fill_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, meter_rect, 1, border_radius=3)

    def _generate_blocks(self):
        blocks = []
        w, h = self.BLOCK_SIZE
        for _ in range(self.NUM_BLOCKS):
            while True:
                x = self.np_random.integers(30, self.WIDTH - w - 30)
                y = self.np_random.integers(50, self.HEIGHT - h - 50)
                new_block = pygame.Rect(x, y, w, h)
                
                start_area = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT/2 - 50, 100, 100)
                if not new_block.colliderect(start_area):
                    blocks.append(new_block)
                    break
        return blocks
        
    def _spawn_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=float)
            self.particles.append({
                'pos': pos.copy().astype(float),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Un-comment the line below to run with a display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bouncing Ball Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(60)
        
    env.close()