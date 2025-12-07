
# Generated: 2025-08-28T05:10:28.942739
# Source Brief: brief_05486.md
# Brief Index: 5486

        
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
        "Controls: Use ↑ and ↓ to steer your ship. Avoid the red obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling racer. Navigate a neon tunnel, dodge obstacles, and reach the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 900  # 30 seconds at 30 FPS

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_OUTLINE = (255, 200, 200)
    COLOR_TRACK = (60, 60, 80)
    COLOR_PARTICLE = (200, 200, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_FINISH_1 = (255, 255, 255)
    COLOR_FINISH_2 = (0, 0, 0)

    # Player
    PLAYER_X_POS = 120
    PLAYER_SIZE = 12
    PLAYER_ACCEL = 0.15
    PLAYER_DRAG = 0.1
    PLAYER_MAX_V = 350.0

    # Track
    TRACK_TOP = 80
    TRACK_BOTTOM = HEIGHT - 80
    TRACK_SPEED = 400.0  # pixels per second
    TRACK_LENGTH = TRACK_SPEED * (MAX_STEPS / FPS)

    # Obstacles
    OBSTACLE_SIZE = 25
    INITIAL_SPAWN_RATE = 0.7  # obstacles per second
    SPAWN_RATE_INCREASE_PER_SEC = 0.05
    NEAR_MISS_DISTANCE = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self.dt = 1.0 / self.FPS

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.time_elapsed = 0.0
        self.track_progress = 0.0
        self.player_y = 0
        self.player_vy = 0.0
        self.obstacles = []
        self.particles = []
        self.current_spawn_rate = 0.0
        self.obstacle_spawn_timer = 0.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.time_elapsed = 0.0
        self.track_progress = 0.0

        self.player_y = self.HEIGHT / 2
        self.player_vy = 0.0

        self.obstacles = []
        self.particles = []
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.obstacle_spawn_timer = 1.0 / self.current_spawn_rate

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self._update_player(movement)
        self._update_obstacles()
        self._update_particles()
        
        self.steps += 1
        self.time_elapsed += self.dt
        self.track_progress += self.TRACK_SPEED * self.dt

        reward, terminated = self._calculate_rewards_and_termination()
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        target_vy = 0
        if movement == 1:  # Up
            target_vy = -self.PLAYER_MAX_V
        elif movement == 2:  # Down
            target_vy = self.PLAYER_MAX_V
        
        # Smooth acceleration and deceleration for better game feel
        self.player_vy += (target_vy - self.player_vy) * self.PLAYER_ACCEL
        if abs(target_vy) < 0.1: # Apply drag if no input
             self.player_vy *= (1 - self.PLAYER_DRAG)

        self.player_y += self.player_vy * self.dt
        self.player_y = np.clip(self.player_y, self.TRACK_TOP + self.PLAYER_SIZE, self.TRACK_BOTTOM - self.PLAYER_SIZE)

    def _update_obstacles(self):
        # Move existing obstacles
        for obs in self.obstacles:
            obs['rect'].x -= self.TRACK_SPEED * self.dt

        # Remove off-screen obstacles to prevent memory leaks
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

        # Spawn new obstacles based on a timer and spawn rate
        self.obstacle_spawn_timer -= self.dt
        if self.obstacle_spawn_timer <= 0:
            obs_y = self.np_random.uniform(self.TRACK_TOP, self.TRACK_BOTTOM - self.OBSTACLE_SIZE)
            new_obs_rect = pygame.Rect(self.WIDTH, obs_y, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
            self.obstacles.append({'rect': new_obs_rect, 'scored': False})
            
            # Increase difficulty over time
            self.current_spawn_rate = self.INITIAL_SPAWN_RATE + self.SPAWN_RATE_INCREASE_PER_SEC * self.time_elapsed
            self.obstacle_spawn_timer = 1.0 / self.current_spawn_rate

    def _update_particles(self):
        # Add speed line particles for visual effect
        if self.np_random.random() < 0.7:
            y = self.np_random.uniform(0, self.HEIGHT)
            self.particles.append({
                'pos': [self.WIDTH, y],
                'vel': [-self.TRACK_SPEED * self.np_random.uniform(1.1, 1.5), 0],
                'life': 1.0,
                'color': self.COLOR_PARTICLE,
                'type': 'line'
            })

        # Update and remove dead particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0] * self.dt
            p['pos'][1] += p['vel'][1] * self.dt
            p['life'] -= self.dt * 2.0
        
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _calculate_rewards_and_termination(self):
        reward = 0.0
        terminated = False

        player_rect = pygame.Rect(self.PLAYER_X_POS - self.PLAYER_SIZE, self.player_y - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)

        for obs in self.obstacles:
            # Collision check
            if player_rect.colliderect(obs['rect']):
                # sfx: explosion
                reward = -100.0
                terminated = True
                self.game_over = True
                self._create_explosion(player_rect.center)
                break
            
            # Reward for passing an obstacle
            if obs['rect'].right < self.PLAYER_X_POS and not obs['scored']:
                obs['scored'] = True
                reward += 1.0
                # sfx: whoosh
                
                # Bonus reward and particle effect for near-misses
                dist_y = abs(player_rect.centery - obs['rect'].centery)
                if dist_y < self.NEAR_MISS_DISTANCE:
                    reward += 0.5
                    self._create_near_miss_effect(obs['rect'].center)

        if terminated: # From collision
            return reward, terminated
        
        # Survival reward for each step
        reward += 0.1

        # Victory condition: Reached the finish line
        if self.track_progress >= self.TRACK_LENGTH:
            time_bonus = 50.0 + 50.0 * max(0, (self.MAX_STEPS / self.FPS) - self.time_elapsed) / (self.MAX_STEPS / self.FPS)
            reward += time_bonus
            # sfx: victory fanfare
            terminated = True
            self.game_over = True
            self.victory = True
        
        # Timeout condition: Ran out of steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return reward, terminated
    
    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(100, 300)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.uniform(0.5, 1.5),
                'color': random.choice([self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_OUTLINE, (255, 255, 0)]),
                'type': 'circle'
            })
            
    def _create_near_miss_effect(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(math.pi / 2, 3 * math.pi / 2) # Sparks fly backwards
            speed = self.np_random.uniform(50, 150)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed - self.TRACK_SPEED, math.sin(angle) * speed],
                'life': self.np_random.uniform(0.2, 0.5),
                'color': self.COLOR_PLAYER_GLOW,
                'type': 'circle'
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw track boundaries and scrolling tick marks for speed effect
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_TOP), (self.WIDTH, self.TRACK_TOP), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_BOTTOM), (self.WIDTH, self.TRACK_BOTTOM), 2)
        
        tick_spacing = 50
        offset = -(self.track_progress % tick_spacing)
        for i in range(self.WIDTH // tick_spacing + 2):
            x = offset + i * tick_spacing
            pygame.draw.line(self.screen, self.COLOR_TRACK, (x, self.TRACK_TOP), (x, self.TRACK_TOP - 5), 1)
            pygame.draw.line(self.screen, self.COLOR_TRACK, (x, self.TRACK_BOTTOM), (x, self.TRACK_BOTTOM + 5), 1)

        # Draw particles with alpha blending for a fading effect
        for p in self.particles:
            alpha = max(0, int(255 * p['life']))
            if p['type'] == 'line':
                end_pos_x = p['pos'][0] + p['vel'][0] * 0.05
                # Pygame doesn't support alpha on basic draw calls, so we skip if alpha is 0
                if alpha > 0:
                    temp_surf = self.screen.copy()
                    line_surf = pygame.Surface((abs(p['pos'][0] - end_pos_x) + 1, 2), pygame.SRCALPHA)
                    pygame.draw.line(line_surf, (*p['color'], alpha), (0, 1), (line_surf.get_width(), 1), 2)
                    self.screen.blit(line_surf, (min(p['pos'][0], end_pos_x), p['pos'][1] - 1))
            elif p['type'] == 'circle' and alpha > 0:
                color = p['color']
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['life'] * 4 + 1), (*color, alpha))

        # Draw finish line when it's approaching
        finish_x = self.WIDTH + self.TRACK_LENGTH - self.track_progress
        if finish_x < self.WIDTH + 50:
            check_size = 20
            for y in range(self.TRACK_TOP, self.TRACK_BOTTOM, check_size):
                # Staggered pattern for checkered flag
                is_even_row = (y - self.TRACK_TOP) // check_size % 2 == 0
                for x_offset in range(0, 40, check_size):
                    is_even_col = x_offset // check_size % 2 == 0
                    color = self.COLOR_FINISH_1 if is_even_row != is_even_col else self.COLOR_FINISH_2
                    pygame.draw.rect(self.screen, color, (finish_x + x_offset, y, check_size, check_size))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obs['rect'], 2)

        # Draw player, unless crashed
        if not (self.game_over and not self.victory):
            px, py = int(self.PLAYER_X_POS), int(self.player_y)
            
            # Glow effect using multiple transparent circles
            for i in range(self.PLAYER_SIZE, 0, -2):
                alpha = 80 * (1 - i / self.PLAYER_SIZE)
                pygame.gfxdraw.filled_circle(self.screen, px, py, i + 5, (*self.COLOR_PLAYER_GLOW, int(alpha)))
            
            # Player triangle with anti-aliasing
            p1 = (px + self.PLAYER_SIZE, py)
            p2 = (px - self.PLAYER_SIZE / 2, py - self.PLAYER_SIZE * 0.866)
            p3 = (px - self.PLAYER_SIZE / 2, py + self.PLAYER_SIZE * 0.866)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

    def _render_ui(self):
        # Time remaining display
        time_left = max(0, (self.MAX_STEPS / self.FPS) - self.time_elapsed)
        time_text = f"TIME: {time_left:.2f}"
        time_surf = self.font_medium.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Score display
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Speed display (flavor text)
        speed_text = f"SPEED: {int(self.TRACK_SPEED)} px/s"
        speed_surf = self.font_small.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (10, self.HEIGHT - speed_surf.get_height() - 10))

        # Game Over / Victory message
        if self.game_over:
            msg = "FINISH!" if self.victory else "CRASHED"
            color = (100, 255, 100) if self.victory else self.COLOR_OBSTACLE
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed,
            "track_progress": self.track_progress,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly to test it.
    human_play = True
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = env.action_space.sample() # Default to random action for AI play
        
        if human_play:
            # Map keyboard inputs to the MultiDiscrete action space
            keys = pygame.key.get_pressed()
            move_action = 0 # no-op
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                move_action = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                move_action = 2
            
            # The brief specifies no effect for space/shift, but they are part of the action space
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [move_action, space_action, shift_action]

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE: # Quit on 'q' or 'escape'
                    running = False

        # Rendering to the display window
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame surfaces use (width, height), while the numpy array obs is (height, width, channels).
        # We need to transpose the axes from (H, W, C) to (W, H, C) for display.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()
    print(f"Game over. Final score: {total_reward:.2f}")