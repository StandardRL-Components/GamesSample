import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball,
    dodging obstacles and hitting targets to score points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control a bouncing ball to hit targets for points while dodging moving obstacles."
    user_guide = "Use the arrow keys (↑↓←→) to apply force to the ball."
    auto_advance = True

    # Constants
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    WIN_SCORE = 500

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (40, 60, 100)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (255, 80, 80)
    COLOR_GREEN = (80, 255, 80)
    COLOR_BLUE = (80, 80, 255)
    COLOR_TEXT = (240, 240, 240)
    
    # Game parameters
    BALL_RADIUS = 12
    BALL_ACCEL = 0.6
    BALL_DRAG = 0.99
    BALL_MAX_VEL = 8
    BALL_RESTITUTION = 0.95
    TARGET_SIZE = 25
    OBSTACLE_WIDTH = 10
    INITIAL_OBSTACLE_SPEED = 2.0
    OBSTACLE_SPEED_INCREASE = 0.2
    
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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.game_over_font = pygame.font.SysFont("monospace", 50, bold=True)

        self._background_surface = self._create_gradient_background()

        # Initialize state variables (will be properly set in reset)
        self.ball_pos = None
        self.ball_vel = None
        self.obstacles = []
        self.targets = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.last_score_milestone = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ball_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float64)
        self.ball_vel = np.array([self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)], dtype=np.float64)
        
        self.obstacles = []
        self.targets = []
        self.particles = []
        
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.last_score_milestone = 0

        self._spawn_initial_targets(3)
        self._spawn_initial_obstacles(4)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self.steps += 1
        
        self._apply_action(movement)
        self._update_ball()
        self._update_obstacles()
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            self._update_particle(p)
            
        reward = 0.1  # Survival reward
        
        hit_target = self._check_target_collisions()
        if hit_target:
            reward += 10
            self.score += 10
            # SFX: Play positive chime
            self._spawn_particles(hit_target.rect.center, hit_target.color, 30)
            self.targets.remove(hit_target)
            self._spawn_target()
            
            if self.score // 100 > self.last_score_milestone:
                self.last_score_milestone = self.score // 100
                self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE
        
        if self._check_obstacle_collisions():
            self.game_over = True
            reward = -50
            # SFX: Play explosion/fail sound
            self._spawn_particles(self.ball_pos, self.COLOR_RED, 50)
            
        terminated = self.game_over
        truncated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Truncated, not terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _apply_action(self, movement):
        if movement == 1: # Up
            self.ball_vel[1] -= self.BALL_ACCEL
        elif movement == 2: # Down
            self.ball_vel[1] += self.BALL_ACCEL
        elif movement == 3: # Left
            self.ball_vel[0] -= self.BALL_ACCEL
        elif movement == 4: # Right
            self.ball_vel[0] += self.BALL_ACCEL
            
        self.ball_vel *= self.BALL_DRAG
        self.ball_vel = np.clip(self.ball_vel, -self.BALL_MAX_VEL, self.BALL_MAX_VEL)

    def _update_ball(self):
        self.ball_pos += self.ball_vel
        
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -self.BALL_RESTITUTION
        elif self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.SCREEN_WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -self.BALL_RESTITUTION

        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -self.BALL_RESTITUTION
        elif self.ball_pos[1] >= self.SCREEN_HEIGHT - self.BALL_RADIUS:
            self.ball_pos[1] = self.SCREEN_HEIGHT - self.BALL_RADIUS
            self.ball_vel[1] *= -self.BALL_RESTITUTION

    def _update_obstacles(self):
        for obstacle in self.obstacles:
            obstacle['rect'].x += obstacle['vel_x']
        
        self.obstacles = [o for o in self.obstacles if o['rect'].right > 0 and o['rect'].left < self.SCREEN_WIDTH]
        while len(self.obstacles) < 4:
            self._spawn_obstacle()
            
    def _check_target_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS,
                                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        for target in self.targets:
            if ball_rect.colliderect(target.rect):
                return target
        return None

    def _check_obstacle_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS,
                                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        for obstacle in self.obstacles:
            if ball_rect.colliderect(obstacle['rect']):
                return True
        return False
        
    def _get_observation(self):
        self.screen.blit(self._background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ball_pos": self.ball_pos.tolist(),
            "ball_vel": self.ball_vel.tolist(),
            "obstacle_speed": self.obstacle_speed,
        }

    def _render_game(self):
        for p in self.particles:
            self._draw_particle(p)
            
        for target in self.targets:
            pygame.draw.rect(self.screen, target.color, target.rect)
            
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, obstacle['color'], obstacle['rect'])
            
        if not self.game_over:
            self._draw_glowing_circle(
                self.screen,
                (int(self.ball_pos[0]), int(self.ball_pos[1])),
                self.BALL_RADIUS,
                self.COLOR_WHITE
            )

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                end_text = self.game_over_font.render("YOU WIN!", True, self.COLOR_GREEN)
            else:
                end_text = self.game_over_font.render("GAME OVER", True, self.COLOR_RED)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_initial_targets(self, count):
        for _ in range(count):
            self._spawn_target()
            
    def _spawn_target(self):
        class Target:
            def __init__(self, rect, color):
                self.rect = rect
                self.color = color

        while True:
            x = self.np_random.integers(self.TARGET_SIZE, self.SCREEN_WIDTH - self.TARGET_SIZE)
            y = self.np_random.integers(self.TARGET_SIZE, self.SCREEN_HEIGHT - self.TARGET_SIZE)
            rect = pygame.Rect(x, y, self.TARGET_SIZE, self.TARGET_SIZE)
            
            # Ensure new target doesn't overlap with existing ones
            is_overlapping = False
            for t in self.targets:
                if rect.colliderect(t.rect):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                color = random.choice([self.COLOR_RED, self.COLOR_GREEN, self.COLOR_BLUE])
                self.targets.append(Target(rect, color))
                break

    def _spawn_initial_obstacles(self, count):
        for _ in range(count):
            self._spawn_obstacle(initial_spawn=True)

    def _spawn_obstacle(self, initial_spawn=False):
        height = self.np_random.integers(50, 150)
        y = self.np_random.integers(0, self.SCREEN_HEIGHT - height)
        
        if self.np_random.choice([True, False]):
            x = self.np_random.integers(-200, -50) if not initial_spawn else self.np_random.integers(0, self.SCREEN_WIDTH)
            vel_x = self.obstacle_speed
        else:
            x = self.np_random.integers(self.SCREEN_WIDTH + 50, self.SCREEN_WIDTH + 200) if not initial_spawn else self.np_random.integers(0, self.SCREEN_WIDTH)
            vel_x = -self.obstacle_speed
            
        rect = pygame.Rect(int(x), int(y), self.OBSTACLE_WIDTH, int(height))
        color = random.choice([self.COLOR_RED, self.COLOR_GREEN, self.COLOR_BLUE])
        self.obstacles.append({'rect': rect, 'vel_x': vel_x, 'color': color})
        
    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'life': self.np_random.integers(20, 40)
            })

    def _update_particle(self, p):
        p['pos'][0] += p['vel'][0]
        p['pos'][1] += p['vel'][1]
        p['vel'][1] += 0.1 # gravity
        p['life'] -= 1

    def _create_gradient_background(self):
        surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(surf, color, (0, y), (self.SCREEN_WIDTH, y))
        return surf

    def _draw_particle(self, p):
        alpha = max(0, min(255, int(255 * (p['life'] / 40))))
        radius = int(p['radius'])
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*p['color'], alpha), (radius, radius), radius)
        self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    def _draw_glowing_circle(self, surface, pos, radius, color):
        for i in range(4, 0, -1):
            alpha = 40 - i * 8
            glow_radius = radius + i * 2
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, (*color, alpha))
            surface.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def close(self):
        pygame.quit()
        
    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup a visible display
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc. depending on your OS
    pygame.display.init()
    pygame.display.set_caption("Bounce Dodge")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        # Get human action
        movement = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        # The other action space dimensions are not used in this game's logic
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the game to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            # You can display a game over message here if you want
            pass

        clock.tick(30)