import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:15:47.453183
# Source Brief: brief_00314.md
# Brief Index: 314
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball to destroy
    colored blocks in a chain reaction, aiming to reach the bottom of the screen
    before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to destroy colored blocks. Nudge the ball on impact to "
        "clear the screen and reach the goal at the bottom before time runs out."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to nudge the ball when it bounces."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # --- Colors ---
    COLOR_BG_TOP = (15, 20, 40)
    COLOR_BG_BOTTOM = (30, 40, 60)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 220, 255)
    COLOR_BLOCKS = {
        'r': (255, 80, 80), 'g': (80, 255, 80), 'b': (80, 120, 255)
    }
    COLOR_BLOCK_BORDERS = {
        'r': (180, 50, 50), 'g': (50, 180, 50), 'b': (50, 80, 180)
    }
    COLOR_OBSTACLE = (100, 100, 110)
    COLOR_OBSTACLE_BORDER = (70, 70, 80)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_UI_SHADOW = (20, 20, 30)

    # --- Game Parameters ---
    BALL_RADIUS = 8
    BALL_START_POS = (WIDTH // 2, BALL_RADIUS * 4)
    BALL_START_VEL = (1.5, 2.0)
    GRAVITY_START = 1.0
    GRAVITY_INCREMENT = 0.1
    SPEED_MULTIPLIER_BUMP = 0.05
    NUDGE_STRENGTH = 1.0
    BLOCK_SIZE = (32, 16)
    INITIAL_BLOCK_COUNT = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Persistent State (across resets for progression) ---
        self.successful_runs = 0

        # --- Initialize State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.gravity = 0.0
        self.speed_multiplier = 1.0
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_trail = deque(maxlen=15)
        self.blocks = []
        self.particles = []
        self.nudge = pygame.Vector2(0, 0)
        self.last_gravity_update_time = 0

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIME_LIMIT_SECONDS
        self.gravity = self.GRAVITY_START
        self.speed_multiplier = 1.0
        self.last_gravity_update_time = self.TIME_LIMIT_SECONDS

        # --- Reset Ball ---
        self.ball_pos = pygame.Vector2(self.BALL_START_POS)
        start_vel_x = self.np_random.choice([-self.BALL_START_VEL[0], self.BALL_START_VEL[0]])
        self.ball_vel = pygame.Vector2(start_vel_x, self.BALL_START_VEL[1])
        self.ball_trail.clear()

        # --- Reset Game Elements ---
        self.particles = []
        self.nudge = pygame.Vector2(0, 0)
        self._generate_blocks()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Update Timer and Difficulty ---
        self.timer -= 1 / self.FPS
        if int(self.timer) % 10 == 9 and int(self.timer) != self.last_gravity_update_time:
             self.gravity = min(3.0, self.gravity + self.GRAVITY_INCREMENT)
             self.last_gravity_update_time = int(self.timer)

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_ball()
        collision_reward = self._handle_collisions()
        reward += collision_reward
        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = False
        win = self.ball_pos.y > self.HEIGHT
        timeout = self.timer <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS

        if win:
            reward += 100  # Goal-oriented reward for winning
            self.score += 1000
            self.successful_runs += 1
            terminated = True
            self.game_over = True
        elif timeout or max_steps_reached:
            reward -= 100  # Penalty for losing
            self.successful_runs = 0 # Reset progression on failure
            terminated = True
            self.game_over = True

        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        self.nudge = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            self.nudge.y = -self.NUDGE_STRENGTH
        elif movement == 2:  # Down
            self.nudge.y = self.NUDGE_STRENGTH
        elif movement == 3:  # Left
            self.nudge.x = -self.NUDGE_STRENGTH
        elif movement == 4:  # Right
            self.nudge.x = self.NUDGE_STRENGTH

    def _update_ball(self):
        self.ball_trail.append(self.ball_pos.copy())
        
        # Apply gravity and speed multiplier
        self.ball_vel.y += self.gravity / self.FPS
        self.ball_pos += self.ball_vel * self.speed_multiplier
        
        # Clamp velocity to prevent extreme speeds
        self.ball_vel.x = np.clip(self.ball_vel.x, -15, 15)
        self.ball_vel.y = np.clip(self.ball_vel.y, -15, 15)

    def _handle_collisions(self):
        reward = 0
        
        # Wall collisions
        bounce = False
        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            bounce = True
        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT)
            bounce = True

        if bounce:
            reward += 0.1 # Reward for any bounce
            self.ball_vel += self.nudge # Apply player nudge on bounce
            self.nudge = pygame.Vector2(0, 0) # Consume nudge

        # Block collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                if block['type'] == 'obstacle':
                    self.ball_vel.y *= -1 
                    self.ball_pos.y += self.ball_vel.y 
                    reward += 0.1
                    continue

                self.blocks.remove(block)
                reward += 1.0 
                self.score += 10
                self.speed_multiplier += self.SPEED_MULTIPLIER_BUMP
                
                self._spawn_particles(block['rect'].center, self.COLOR_BLOCKS[block['color_key']], 20)

                self.ball_vel.y *= -1
                self.ball_vel += self.nudge 
                self.nudge = pygame.Vector2(0, 0) 
                
                if self.score % 50 == 0 and self.score > 0:
                    reward += 5.0
                    self.score += 50
                
                break 

        return reward

    def _generate_blocks(self):
        self.blocks.clear()
        block_count = int(self.INITIAL_BLOCK_COUNT * (1.1 ** self.successful_runs))
        obstacle_count = 5 if self.successful_runs >= 3 else 0
        
        w, h = self.BLOCK_SIZE
        grid_w, grid_h = self.WIDTH // w, self.HEIGHT // h
        possible_positions = []
        for gx in range(grid_w):
            for gy in range(3, grid_h - 3):
                possible_positions.append((gx, gy))

        self.np_random.shuffle(possible_positions)
        
        # Place normal blocks
        for i in range(min(len(possible_positions), block_count)):
            gx, gy = possible_positions.pop(0)
            x, y = gx * w, gy * h
            color_key = self.np_random.choice(list(self.COLOR_BLOCKS.keys()))
            self.blocks.append({
                'rect': pygame.Rect(x, y, w, h),
                'color_key': color_key,
                'type': 'normal'
            })
        
        # Place obstacles
        for i in range(min(len(possible_positions), obstacle_count)):
            gx, gy = possible_positions.pop(0)
            x, y = gx * w, gy * h
            self.blocks.append({
                'rect': pygame.Rect(x, y, w, h),
                'color_key': None,
                'type': 'obstacle'
            })

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 31)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": round(self.timer, 2),
            "speed_multiplier": round(self.speed_multiplier, 2),
            "successful_runs": self.successful_runs
        }

    def _render_all(self):
        self._render_background()
        self._render_blocks()
        self._render_particles()
        self._render_ball()
        self._render_ui()

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_blocks(self):
        for block in self.blocks:
            if block['type'] == 'obstacle':
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, block['rect'])
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_BORDER, block['rect'], 2)
            else:
                pygame.draw.rect(self.screen, self.COLOR_BLOCKS[block['color_key']], block['rect'])
                pygame.draw.rect(self.screen, self.COLOR_BLOCK_BORDERS[block['color_key']], block['rect'], 2)

    def _render_ball(self):
        # Trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
            radius = int(self.BALL_RADIUS * (i / len(self.ball_trail)))
            if radius > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(pos.x), int(pos.y), radius,
                    (self.COLOR_BALL_GLOW[0], self.COLOR_BALL_GLOW[1], self.COLOR_BALL_GLOW[2], alpha)
                )
        
        # Glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_alpha = 90
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos.x), int(self.ball_pos.y), glow_radius,
            (self.COLOR_BALL_GLOW[0], self.COLOR_BALL_GLOW[1], self.COLOR_BALL_GLOW[2], glow_alpha)
        )
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for p in self.particles:
            ratio = p['lifespan'] / p['max_lifespan']
            radius = int(ratio * 4)
            if radius > 0:
                alpha = int(ratio * 255)
                color_with_alpha = (p['color'][0], p['color'][1], p['color'][2], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, color_with_alpha)

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow_color):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        speed_text = f"SPEED: {self.speed_multiplier:.2f}x"
        draw_text(speed_text, self.font_small, self.COLOR_UI_TEXT, (10, 10), self.COLOR_UI_SHADOW)

        timer_text = f"{max(0, self.timer):.1f}"
        text_surf = self.font_large.render(timer_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 10, 10))
        shadow_surf = self.font_large.render(timer_text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)
        
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_small, self.COLOR_UI_TEXT, (10, 35), self.COLOR_UI_SHADOW)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    running = True
    total_reward = 0
    
    # Create a display window
    pygame.display.set_caption("Chain Reaction Bouncer")
    game_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        env.clock.tick(GameEnv.FPS)

    pygame.quit()