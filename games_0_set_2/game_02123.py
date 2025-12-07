
# Generated: 2025-08-28T03:47:35.488441
# Source Brief: brief_02123.md
# Brief Index: 2123

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Press SPACE in time with the beat markers to accelerate."

    # Must be a short, user-facing description of the game:
    game_description = "A synthwave-themed rhythm-racing game. Hit the beats to build speed and race to the finish line. Don't let your accuracy drop!"

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto-advance logic

        # Colors (Synthwave Palette)
        self.COLOR_BG = (13, 10, 33) # Dark Navy
        self.COLOR_GRID = (38, 30, 81) # Muted Purple
        self.COLOR_PLAYER = (0, 255, 255) # Bright Cyan
        self.COLOR_SUCCESS = (0, 255, 127) # Bright Green
        self.COLOR_FAIL = (255, 64, 64) # Bright Red
        self.COLOR_BEAT = (255, 255, 0) # Bright Yellow
        self.COLOR_OBSTACLE = (255, 0, 127) # Bright Magenta
        self.COLOR_UI_TEXT = (220, 220, 220) # Off-white
        self.COLOR_TRACK = (100, 100, 150)

        # Game parameters
        self.TRACK_LENGTH = 50000
        self.MAX_STEPS = 5000
        self.PLAYER_BASE_X = 100
        self.HIT_TARGET_X = 150
        self.HIT_WINDOW = 30 # pixels on either side of the target line
        self.TRACK_Y_TOP = 180
        self.TRACK_Y_BOTTOM = 220
        self.MIN_SPEED = 1.0
        self.MAX_SPEED = 20.0
        self.INITIAL_SPEED = 3.0
        self.INITIAL_BEAT_RATE = 1.0 # Hz
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 22)
            self.font_large = pygame.font.Font(None, 30)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.distance_traveled = 0.0
        self.speed = 0.0
        self.total_beats_processed = 0
        self.successful_hits = 0
        self.accuracy = 1.0
        self.last_space_held = False
        self.beat_spawn_timer = 0.0
        self.beat_spawn_rate = 0.0
        self.required_finish_accuracy = 0.8
        self.grid_offset = 0.0
        self.player_y = 0.0
        self.player_bob_angle = 0.0
        self.beats = []
        self.obstacles = []
        self.particles = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.distance_traveled = 0.0
        self.speed = self.INITIAL_SPEED
        self.total_beats_processed = 0
        self.successful_hits = 0
        self.accuracy = 1.0
        self.last_space_held = False
        self.beat_spawn_rate = self.INITIAL_BEAT_RATE
        self.beat_spawn_timer = 0.0
        self.required_finish_accuracy = 0.8
        self.grid_offset = 0.0
        
        self.player_y = (self.TRACK_Y_TOP + self.TRACK_Y_BOTTOM) / 2
        self.player_bob_angle = 0.0

        self.beats = []
        self.obstacles = []
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # --- Action Handling ---
        # movement = action[0] # Unused
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused
        
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        # --- Game Logic Update ---
        self.steps += 1
        self.speed = max(self.MIN_SPEED, self.speed) # Ensure we always move
        self.distance_traveled += self.speed
        
        self.player_bob_angle = (self.player_bob_angle + 0.1) % (2 * math.pi)
        self.player_y = ((self.TRACK_Y_TOP + self.TRACK_Y_BOTTOM) / 2) + math.sin(self.player_bob_angle) * 2

        # --- Beat & Obstacle Spawning ---
        self.beat_spawn_timer += self.beat_spawn_rate / self.FPS
        if self.beat_spawn_timer >= 1.0:
            self.beat_spawn_timer -= 1.0
            self.beats.append({
                "x": self.WIDTH + 50,
                "y": (self.TRACK_Y_TOP + self.TRACK_Y_BOTTOM) / 2,
                "hit": False,
                "processed": False
            })
            if self.np_random.random() < 0.15:
                obstacle_y = self.np_random.choice([self.TRACK_Y_TOP + 5, self.TRACK_Y_BOTTOM - 25])
                self.obstacles.append(pygame.Rect(self.WIDTH + 150, obstacle_y, 30, 20))

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.beat_spawn_rate = min(3.0, self.beat_spawn_rate + 0.05)
            self.required_finish_accuracy = min(0.95, self.required_finish_accuracy + 0.02)
            
        # --- Update Entities ---
        for beat in self.beats: beat["x"] -= self.speed
        for obstacle in self.obstacles: obstacle.x -= self.speed

        # --- Handle Player Input (Space Press) ---
        if space_pressed:
            closest_beat, min_dist = None, float('inf')
            for beat in self.beats:
                if not beat["hit"]:
                    dist = abs(beat["x"] - self.HIT_TARGET_X)
                    if dist < min_dist:
                        min_dist, closest_beat = dist, beat
            
            if closest_beat and min_dist <= self.HIT_WINDOW:
                closest_beat["hit"] = closest_beat["processed"] = True
                self.successful_hits += 1
                self.total_beats_processed += 1
                reward += 1.0
                self.score += 10
                self.speed = min(self.MAX_SPEED, self.speed + 1.0)
                # sfx: positive hit sound
                self._create_particles(self.HIT_TARGET_X, closest_beat["y"], self.COLOR_SUCCESS, 20)
            else:
                reward -= 0.1
                self.speed = max(self.MIN_SPEED, self.speed - 0.2)
                # sfx: whiff sound
        
        # --- Process Passed Beats (Misses) ---
        for beat in self.beats:
            if not beat["processed"] and beat["x"] < self.HIT_TARGET_X - self.HIT_WINDOW:
                beat["processed"] = True
                self.total_beats_processed += 1
                reward -= 1.0
                self.score -= 5
                self.speed = max(self.MIN_SPEED, self.speed * 0.8)
                # sfx: negative miss sound
                self._create_particles(beat["x"], beat["y"], self.COLOR_FAIL, 10)
        
        # --- Handle Obstacle Collisions ---
        player_rect = pygame.Rect(self.PLAYER_BASE_X - 10, self.player_y - 10, 20, 20)
        collided_obstacle = player_rect.collidelist(self.obstacles)
        if collided_obstacle != -1:
            reward -= 2.0
            self.score -= 20
            self.speed *= 0.5
            self._create_particles(player_rect.centerx, player_rect.centery, self.COLOR_OBSTACLE, 30)
            self.obstacles.pop(collided_obstacle)
            # sfx: crash sound
                
        # --- Update Accuracy ---
        if self.total_beats_processed > 0:
            self.accuracy = self.successful_hits / self.total_beats_processed
        
        # --- Cleanup & Particle Update ---
        self.beats = [b for b in self.beats if b["x"] > -50]
        self.obstacles = [o for o in self.obstacles if o.right > 0]
        for p in self.particles[:]:
            p['x'] += p['vx']; p['y'] += p['vy']; p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)

        # --- Termination Conditions ---
        terminated = False
        if self.distance_traveled >= self.TRACK_LENGTH:
            terminated = True
            reward += 100.0 if self.accuracy >= self.required_finish_accuracy else -50.0
            self.score += 1000 if self.accuracy >= self.required_finish_accuracy else 0
        elif self.total_beats_processed > 20 and self.accuracy < 0.5:
            terminated = True
            reward -= 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 50.0
            
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_track()
        self._render_obstacles()
        self._render_beats()
        self._render_hit_zone()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "accuracy": self.accuracy,
            "distance_remaining": max(0, self.TRACK_LENGTH - self.distance_traveled),
        }

    def _render_background(self):
        self.grid_offset = (self.grid_offset - self.speed * 0.5) % 50
        for i in range(self.WIDTH // 50 + 2):
            x = int(i * 50 + self.grid_offset)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for i in range(self.HEIGHT // 50 + 1):
            y = i * 50
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_track(self):
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP), (self.WIDTH, self.TRACK_Y_TOP), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_BOTTOM), (self.WIDTH, self.TRACK_Y_BOTTOM), 2)
        
    def _render_hit_zone(self):
        hit_zone_surface = pygame.Surface((self.HIT_WINDOW * 2, self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP), pygame.SRCALPHA)
        hit_zone_surface.fill((*self.COLOR_SUCCESS, 30))
        self.screen.blit(hit_zone_surface, (self.HIT_TARGET_X - self.HIT_WINDOW, self.TRACK_Y_TOP))
        pygame.draw.line(self.screen, self.COLOR_SUCCESS, (self.HIT_TARGET_X, self.TRACK_Y_TOP - 10), (self.HIT_TARGET_X, self.TRACK_Y_BOTTOM + 10), 2)

    def _render_beats(self):
        for beat in self.beats:
            pulse = abs(math.sin(self.steps * 0.3)) * 3
            radius = int(10 + pulse)
            pygame.gfxdraw.filled_circle(self.screen, int(beat["x"]), int(beat["y"]), radius, self.COLOR_BEAT)
            pygame.gfxdraw.aacircle(self.screen, int(beat["x"]), int(beat["y"]), radius, self.COLOR_BEAT)

    def _render_obstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)
            highlight_rect = obstacle.copy(); highlight_rect.height = 4
            pygame.draw.rect(self.screen, (255, 128, 200), highlight_rect)

    def _render_player(self):
        x, y = int(self.PLAYER_BASE_X), int(self.player_y)
        acc_factor = max(0, (self.accuracy - 0.5) * 2)
        player_color = self._interpolate_color(self.COLOR_FAIL, self.COLOR_SUCCESS, acc_factor)
        
        glow_size = 20 + abs(math.sin(self.steps * 0.2)) * 5
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*player_color, 50), (glow_size, glow_size), int(glow_size))
        self.screen.blit(glow_surf, (x - glow_size, y - glow_size))

        points = [(x + 15, y), (x - 10, y - 12), (x - 5, y), (x - 10, y + 12)]
        pygame.gfxdraw.aapolygon(self.screen, points, player_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, player_color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                radius = max(0, int(p['life'] / p['max_life'] * 5))
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, (*p['color'], alpha))

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        acc_factor = max(0, (self.accuracy - 0.5) * 2)
        acc_color = self._interpolate_color(self.COLOR_FAIL, self.COLOR_SUCCESS, acc_factor)
        acc_text = self.font_large.render(f"ACC: {self.accuracy:.1%}", True, acc_color)
        self.screen.blit(acc_text, (self.WIDTH - acc_text.get_width() - 10, 10))

        progress = min(1.0, self.distance_traveled / self.TRACK_LENGTH)
        bar_width, bar_height = self.WIDTH - 20, 10
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.HEIGHT - 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SUCCESS, (10, self.HEIGHT - 20, int(bar_width * progress), bar_height))
        player_icon_pos = int(10 + bar_width * progress)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (player_icon_pos, self.HEIGHT - 15), 8)

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            max_life = self.np_random.integers(15, 30)
            self.particles.append({
                'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': max_life, 'max_life': max_life, 'color': color
            })

    def _interpolate_color(self, c1, c2, factor):
        f = max(0.0, min(1.0, factor))
        return tuple(int(a + (b - a) * f) for a, b in zip(c1, c2))

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    keys = {'space': False}

    while running:
        if terminated:
            print(f"Game Over! Final Score: {info.get('score', 0)}. Accuracy: {info.get('accuracy', 0):.1%}")
            obs, info = env.reset()
            terminated = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: keys['space'] = True
                elif event.key == pygame.K_r: terminated = True # Force reset
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: keys['space'] = False
        
        action = [0, 1 if keys['space'] else 0, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()