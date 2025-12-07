import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to move. Hold Space to move faster, hold Shift to move slower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race a sleek neon line against time. Dodge scrolling obstacles to survive and score points. Reach the end to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_TRACK = (40, 20, 80)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    OBSTACLE_COLORS = [(255, 0, 128), (255, 128, 0), (255, 255, 0)] # Fast, Medium, Slow
    COLOR_PARTICLE = (255, 255, 200)
    COLOR_TEXT = (220, 220, 220)
    COLOR_FINISH = (255, 255, 255)

    # Player
    PLAYER_X = SCREEN_WIDTH // 4
    PLAYER_WIDTH = 8
    PLAYER_HEIGHT = 40
    PLAYER_ACCEL = 1.0
    PLAYER_FRICTION = 0.85
    PLAYER_MAX_VY = 15.0
    
    # Track
    TRACK_Y_TOP = 50
    TRACK_Y_BOTTOM = SCREEN_HEIGHT - 50
    
    # Obstacles
    OBSTACLE_MIN_WIDTH = 20
    OBSTACLE_MAX_WIDTH = 80
    OBSTACLE_HEIGHT = 20
    
    class Particle:
        def __init__(self, pos, vel, radius, life, color):
            self.pos = list(pos)
            self.vel = list(vel)
            self.radius = radius
            self.life = life
            self.max_life = life
            self.color = color

        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.life -= 1
            self.radius -= 0.1
            return self.life > 0 and self.radius > 0

        def draw(self, surface):
            if self.life > 0:
                alpha = int(255 * (self.life / self.max_life))
                # Create a temporary surface for alpha blending if gfxdraw doesn't support it directly
                # For simplicity, we assume direct alpha support or that the visual effect is acceptable.
                # A more robust way would be to draw on a separate surface with per-pixel alpha.
                try:
                    pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), max(0, int(self.radius)), self.color + (alpha,))
                except TypeError: # Fallback if color with alpha fails
                    pygame.draw.circle(surface, self.color, (int(self.pos[0]), int(self.pos[1])), max(0, int(self.radius)))


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # self.reset() is called by the environment wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_y = (self.TRACK_Y_TOP + self.TRACK_Y_BOTTOM) / 2
        self.player_vy = 0
        
        self.obstacles = []
        self.particles = []
        
        self.last_spawn_step = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = self.game_over

        if not terminated:
            # --- Action Handling ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            speed_mod = 1.0
            if space_held:
                speed_mod = 1.5  # Boost
            elif shift_held:
                speed_mod = 0.5  # Brake

            if movement == 1:  # Up
                self.player_vy -= self.PLAYER_ACCEL * speed_mod
            elif movement == 2:  # Down
                self.player_vy += self.PLAYER_ACCEL * speed_mod

            # --- Physics & Game Logic Update ---
            self.player_vy *= self.PLAYER_FRICTION
            self.player_vy = max(-self.PLAYER_MAX_VY, min(self.PLAYER_MAX_VY, self.player_vy))
            self.player_y += self.player_vy
            
            # Clamp player position
            player_top = self.player_y - self.PLAYER_HEIGHT / 2
            player_bottom = self.player_y + self.PLAYER_HEIGHT / 2
            if player_top < self.TRACK_Y_TOP:
                self.player_y = self.TRACK_Y_TOP + self.PLAYER_HEIGHT / 2
                self.player_vy = 0
            if player_bottom > self.TRACK_Y_BOTTOM:
                self.player_y = self.TRACK_Y_BOTTOM - self.PLAYER_HEIGHT / 2
                self.player_vy = 0

            # Add player trail particles
            if self.np_random.random() < 0.7:
                trail_vel_x = self.np_random.uniform(0.5, 1.5)
                trail_vel_y = self.np_random.uniform(-0.5, 0.5)
                self.particles.append(self.Particle(
                    (self.PLAYER_X, self.player_y), [trail_vel_x, trail_vel_y],
                    self.np_random.uniform(2, 4), 20, self.COLOR_PLAYER_GLOW
                ))

            # --- Obstacle Management ---
            self._update_obstacles()
            self._spawn_obstacles()

            # --- Particle Update ---
            self.particles = [p for p in self.particles if p.update()]

            # --- Collision & Reward Calculation ---
            reward = 0.1  # Survival reward
            player_rect = pygame.Rect(self.PLAYER_X - self.PLAYER_WIDTH / 2, self.player_y - self.PLAYER_HEIGHT / 2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
            
            for obs in self.obstacles:
                if not obs['passed'] and obs['rect'].right < self.PLAYER_X:
                    obs['passed'] = True
                    reward += 1.0  # Dodge reward
                    self.score += 10
                    
                if player_rect.colliderect(obs['rect']):
                    terminated = True
                    self.game_over = True
                    reward = -100  # Collision penalty
                    self.score -= 1000
                    self._create_explosion(player_rect.center, 50)
                    break
            
            # --- Step & Termination Update ---
            self.steps += 1
            if self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True # Treat as game over to stop updates
                reward = 100  # Victory reward
                self.score += 5000
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_obstacles(self):
        # Difficulty scaling for spawn rate
        progress = min(1.0, self.steps / 500.0)
        current_spawn_period = int(50 - progress * 25)
        
        if self.steps > self.last_spawn_step + current_spawn_period:
            self.last_spawn_step = self.steps
            
            # Difficulty scaling for speed
            speed_multiplier = 1.0 + (self.steps / 100.0) * 0.01
            base_speed = self.np_random.uniform(2.0, 5.0) * speed_multiplier
            
            width = self.np_random.integers(self.OBSTACLE_MIN_WIDTH, self.OBSTACLE_MAX_WIDTH)
            
            # Ensure obstacle doesn't spawn right on top of player
            while True:
                y = self.np_random.uniform(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM - self.OBSTACLE_HEIGHT)
                if abs(y - (self.player_y - self.OBSTACLE_HEIGHT / 2)) > self.PLAYER_HEIGHT * 1.5:
                    break

            color_index = 0
            if base_speed < 3.0 * speed_multiplier: color_index = 2 # Slow
            elif base_speed < 4.0 * speed_multiplier: color_index = 1 # Medium
            
            self.obstacles.append({
                'rect': pygame.Rect(self.SCREEN_WIDTH, y, width, self.OBSTACLE_HEIGHT),
                'speed': base_speed,
                'color': self.OBSTACLE_COLORS[color_index],
                'passed': False
            })

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['rect'].x -= obs['speed']
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

    def _create_explosion(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 8)
            life = self.np_random.integers(20, 50)
            self.particles.append(self.Particle(pos, vel, radius, life, self.COLOR_PARTICLE))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw track boundaries with glow effect
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP), (self.SCREEN_WIDTH, self.TRACK_Y_TOP), 3)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_BOTTOM), (self.SCREEN_WIDTH, self.TRACK_Y_BOTTOM), 3)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, obs['color'], obs['rect'])
            # Add a slight inner glow
            inner_rect = obs['rect'].inflate(-4, -4)
            pygame.draw.rect(self.screen, tuple(min(255, c + 50) for c in obs['color']), inner_rect, 1)

        # Draw player with glow
        if not (self.game_over and self.steps < self.MAX_STEPS):
            player_rect = pygame.Rect(0, 0, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
            player_rect.center = (self.PLAYER_X, self.player_y)
            
            # Glow
            glow_rect = player_rect.inflate(self.PLAYER_WIDTH, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, glow_rect, border_radius=4)
            
            # Core
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        steps_text = self.font_small.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                msg = "FINISH!"
                color = self.COLOR_FINISH
            else:
                msg = "GAME OVER"
                color = self.OBSTACLE_COLORS[0]
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the headless environment variable
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Line Racer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
            
        # Get key states for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        else:
            movement = 0
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()