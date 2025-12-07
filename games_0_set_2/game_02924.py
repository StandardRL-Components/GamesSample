
# Generated: 2025-08-27T21:50:46.198933
# Source Brief: brief_02924.md
# Brief Index: 2924

        
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


# It's good practice for headless environments to prevent display pop-ups
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the net and catch fish."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch 25 fish before the 60-second timer runs out in this fast-paced arcade fishing game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.WIN_SCORE = 25
        self.GAME_TIME_LIMIT = 60.0  # seconds

        # Player
        self.PLAYER_SPEED = 5
        self.PLAYER_RADIUS = 20

        # Fish
        self.NUM_FISH = 15
        self.FISH_SIZE = (20, 10)
        self.INITIAL_FISH_SPEED = 1.5
        self.FISH_SPEED_INCREASE = 0.75 # Speed increase per 10 fish

        # Colors
        self.COLOR_WATER_BG = (20, 40, 80)
        self.COLOR_WATER_FG = (30, 60, 120)
        self.COLOR_NET_OUTLINE = (50, 255, 50)
        self.COLOR_NET_FILL = (50, 255, 50, 50)  # RGBA
        self.COLOR_FISH = (255, 150, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.COLOR_TIMER_WARN = (255, 50, 50)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.fishes = None
        self.score = None
        self.time_remaining = None
        self.steps = None
        self.game_over = None
        self.fish_base_speed = None
        self.difficulty_fish_counter = None
        self.particles = None
        self.np_random = None
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.fish_base_speed = self.INITIAL_FISH_SPEED
        self.difficulty_fish_counter = 0
        self.fishes = [self._spawn_fish() for _ in range(self.NUM_FISH)]

        self.score = 0
        self.time_remaining = self.GAME_TIME_LIMIT
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _spawn_fish(self, respawn=False):
        if respawn:
            edge = self.np_random.integers(4)
            if edge == 0: pos = pygame.math.Vector2(self.np_random.random() * self.WIDTH, -self.FISH_SIZE[1])
            elif edge == 1: pos = pygame.math.Vector2(self.np_random.random() * self.WIDTH, self.HEIGHT + self.FISH_SIZE[1])
            elif edge == 2: pos = pygame.math.Vector2(-self.FISH_SIZE[0], self.np_random.random() * self.HEIGHT)
            else: pos = pygame.math.Vector2(self.WIDTH + self.FISH_SIZE[0], self.np_random.random() * self.HEIGHT)
        else:
            pos = pygame.math.Vector2(self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT)

        angle = self.np_random.random() * 2 * math.pi
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * self.fish_base_speed
        
        return {"pos": pos, "vel": vel, "rect": pygame.Rect(pos.x, pos.y, self.FISH_SIZE[0], self.FISH_SIZE[1])}

    def _get_closest_fish_dist(self):
        if not self.fishes: return float('inf')
        return min(self.player_pos.distance_to(fish["pos"]) for fish in self.fishes)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        movement = action[0]
        
        dist_before_move = self._get_closest_fish_dist()

        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        dist_after_move = self._get_closest_fish_dist()
        
        for fish in self.fishes:
            fish["pos"] += fish["vel"]
            if not (0 < fish["pos"].x < self.WIDTH - self.FISH_SIZE[0]): fish["vel"].x *= -1
            if not (0 < fish["pos"].y < self.HEIGHT - self.FISH_SIZE[1]): fish["vel"].y *= -1
            fish["rect"].topleft = fish["pos"]

        if self.np_random.random() < 0.2:
            self._create_particles(1, pygame.math.Vector2(self.np_random.random() * self.WIDTH, self.HEIGHT + 10), 'bubble')
        self.particles = [p for p in self.particles if self._update_particle(p)]

        reward = 0
        fish_caught_this_step = False
        
        for i, fish in enumerate(self.fishes):
            closest_point = pygame.math.Vector2(np.clip(self.player_pos.x, fish["rect"].left, fish["rect"].right), np.clip(self.player_pos.y, fish["rect"].top, fish["rect"].bottom))
            if self.player_pos.distance_to(closest_point) < self.PLAYER_RADIUS:
                # SFX: catch_sound.play()
                self.score += 1
                self.difficulty_fish_counter += 1
                reward += 1.0
                fish_caught_this_step = True
                
                self._create_particles(20, fish["pos"] + pygame.math.Vector2(self.FISH_SIZE[0]/2, self.FISH_SIZE[1]/2), 'catch')
                
                if self.difficulty_fish_counter >= 10:
                    self.difficulty_fish_counter = 0
                    self.fish_base_speed += self.FISH_SPEED_INCREASE
                
                self.fishes[i] = self._spawn_fish(respawn=True)
        
        if not fish_caught_this_step:
            reward += (dist_before_move - dist_after_move) * 0.01

        terminated = (self.score >= self.WIN_SCORE) or (self.time_remaining <= 0)
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE: reward += 100
            else: reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, count, pos, p_type):
        for _ in range(count):
            if p_type == 'bubble':
                vel = pygame.math.Vector2(self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-1.5, -0.5))
                radius = self.np_random.integers(2, 6)
                lifetime = self.np_random.integers(120, 240)
                color = (100, 150, 255)
            elif p_type == 'catch':
                angle = self.np_random.random() * 2 * math.pi
                speed = self.np_random.uniform(1, 4)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                radius = self.np_random.integers(2, 5)
                lifetime = self.np_random.integers(20, 40)
                color = random.choice([(255, 255, 100), (255, 255, 255)])
            
            self.particles.append({"pos": pos.copy(), "vel": vel, "radius": radius, "lifetime": lifetime, "max_life": lifetime, "color": color, "type": p_type})

    def _update_particle(self, p):
        p["pos"] += p["vel"]
        p["lifetime"] -= 1
        if p["vel"].length() > 0.1: p["vel"] *= 0.95
        return p["lifetime"] > 0

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_WATER_BG)
        for i in range(10): pygame.draw.rect(self.screen, self.COLOR_WATER_FG, (i * 64, 0, 32, self.HEIGHT))
        
        for p in self.particles:
            if p["type"] == 'bubble':
                alpha = int(255 * (p['lifetime'] / p['max_life']))
                color = p['color'] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), (200, 220, 255, alpha))

        for fish in self.fishes:
            pygame.draw.rect(self.screen, self.COLOR_FISH, fish["rect"], border_radius=3)

        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        s = pygame.Surface((self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_NET_FILL, (self.PLAYER_RADIUS, self.PLAYER_RADIUS), self.PLAYER_RADIUS)
        self.screen.blit(s, (player_pos_int[0] - self.PLAYER_RADIUS, player_pos_int[1] - self.PLAYER_RADIUS))
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_NET_OUTLINE)
        
        for p in self.particles:
            if p["type"] == 'catch':
                alpha = 255 * (p['lifetime'] / p['max_life'])
                size = p['radius'] * (p['lifetime'] / p['max_life'])
                color = p['color'] + (int(alpha),)
                pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), int(size))

        self._render_text(f"SCORE: {self.score}", self.font_large, self.COLOR_TEXT, (10, 10))
        time_color = self.COLOR_TEXT if self.time_remaining > 10 else self.COLOR_TIMER_WARN
        time_str = f"TIME: {max(0, math.ceil(self.time_remaining))}"
        self._render_text(time_str, self.font_large, time_color, (self.WIDTH - 150, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            self._render_text(msg, self.font_large, self.COLOR_TEXT, (self.WIDTH/2 - self.font_large.size(msg)[0]/2, self.HEIGHT/2 - 50))
            final_score_msg = f"Final Score: {self.score}"
            self._render_text(final_score_msg, self.font_small, self.COLOR_TEXT, (self.WIDTH/2 - self.font_small.size(final_score_msg)[0]/2, self.HEIGHT/2))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_remaining": self.time_remaining}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # To see the game, we need a display
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    
    running = True
    while running:
        movement = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

    env.close()