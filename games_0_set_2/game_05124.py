
# Generated: 2025-08-28T04:02:49.226201
# Source Brief: brief_05124.md
# Brief Index: 5124

        
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
    user_guide = "Controls: ↑/Space to jump, →/Shift to dash. Avoid red obstacles and dash through green ones."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced auto-runner. Jump and dash to survive an onslaught of obstacles for as long as you can."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto_advance=True

        # Colors
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_OBSTACLE = (220, 50, 50)
        self.COLOR_BREAKABLE = (50, 220, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_DASH_TRAIL = (255, 255, 150)

        # Player Physics
        self.PLAYER_X = 100
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 40
        self.GROUND_Y = self.HEIGHT - 50
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.DASH_DURATION = 8 # frames
        self.DASH_COOLDOWN = 15 # frames

        # Game Mechanics
        self.INITIAL_SPEED = 4.0
        self.SPEED_INCREASE_RATE = 0.1
        self.SPEED_INCREASE_INTERVAL = 50
        self.INITIAL_SPAWN_PROB = 0.02
        self.SPAWN_PROB_INCREASE_RATE = 0.005
        self.SPAWN_PROB_INCREASE_INTERVAL = 100
        self.MAX_STEPS = 1500
        self.MAX_COLLISIONS = 2

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # --- State Variables (initialized in reset) ---
        self.player_pos = None
        self.player_vel_y = None
        self.on_ground = None
        self.is_dashing = None
        self.dash_timer = None
        self.dash_cooldown_timer = None
        self.dash_trail = None
        self.obstacles = None
        self.particles = None
        self.game_speed = None
        self.spawn_prob = None
        self.steps = None
        self.score = None
        self.collision_count = None
        self.game_over = None
        self.np_random = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.player_pos = pygame.Vector2(self.PLAYER_X, self.GROUND_Y)
        self.player_vel_y = 0
        self.on_ground = True
        self.is_dashing = False
        self.dash_timer = 0
        self.dash_cooldown_timer = 0
        self.dash_trail = []

        self.obstacles = []
        self.particles = []

        self.game_speed = self.INITIAL_SPEED
        self.spawn_prob = self.INITIAL_SPAWN_PROB

        self.steps = 0
        self.score = 0
        self.collision_count = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # JUMP
        if (movement == 1 or space_held) and self.on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump sound

        # DASH
        if (movement == 4 or shift_held) and not self.is_dashing and self.dash_cooldown_timer == 0:
            self.is_dashing = True
            self.dash_timer = self.DASH_DURATION
            self.dash_cooldown_timer = self.DASH_COOLDOWN
            # sfx: dash sound
            self._create_particles(self.player_pos + (self.PLAYER_WIDTH / 2, -self.PLAYER_HEIGHT / 2), 20, self.COLOR_DASH_TRAIL, 2, 5)

        reward = self._update_player(action)
        self._update_obstacles()
        collision_reward = self._handle_collisions()
        reward += collision_reward
        self._update_particles()
        
        if self.steps > 0:
            if self.steps % self.SPEED_INCREASE_INTERVAL == 0:
                self.game_speed += self.SPEED_INCREASE_RATE
            if self.steps % self.SPAWN_PROB_INCREASE_INTERVAL == 0:
                self.spawn_prob = min(0.1, self.spawn_prob + self.SPAWN_PROB_INCREASE_RATE)

        self.steps += 1
        self.score += 0.1 # Survival score
        reward += 0.1 # Survival reward

        terminated = self.collision_count >= self.MAX_COLLISIONS or self.steps >= 1000
        if terminated:
            self.game_over = True
            if self.collision_count >= self.MAX_COLLISIONS:
                reward = -100 # Penalty for losing
            else: # Reached end
                reward = 50 # Bonus for winning
                self.score += 50
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.is_dashing:
            self.dash_timer -= 1
            if self.dash_timer <= 0:
                self.is_dashing = False
            trail_rect = pygame.Rect(self.player_pos.x, self.player_pos.y - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
            self.dash_trail.append({'rect': trail_rect, 'life': 5})
        
        if self.dash_cooldown_timer > 0:
            self.dash_cooldown_timer -= 1

        self.dash_trail = [t for t in self.dash_trail if t['life'] > 0]
        for t in self.dash_trail: t['life'] -= 1

        if not self.on_ground:
            self.player_vel_y += self.GRAVITY
            self.player_pos.y += self.player_vel_y

        if self.player_pos.y > self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel_y = 0
            if not self.on_ground: # Just landed
                 self._create_particles(pygame.Vector2(self.player_pos.x + self.PLAYER_WIDTH/2, self.GROUND_Y), 5, (200,200,200), 1, 2)
                 # sfx: land sound
            self.on_ground = True
        
        reward = 0
        is_acting = movement != 0 or space_held or shift_held
        if not is_acting:
            for obs in self.obstacles:
                if self.PLAYER_X < obs['rect'].right < self.PLAYER_X + 300 and obs['rect'].top < self.GROUND_Y:
                    reward -= 0.2
                    break
        return reward

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['rect'].x -= self.game_speed
        
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

        if self.np_random.random() < self.spawn_prob:
            self._spawn_obstacle()

    def _spawn_obstacle(self):
        if self.obstacles and self.obstacles[-1]['rect'].right > self.WIDTH - 150:
            return

        obs_height = self.np_random.integers(30, 100)
        obs_width = self.np_random.integers(20, 50)
        
        on_ground = self.np_random.random() > 0.4
        y_pos = self.GROUND_Y - obs_height if on_ground else self.GROUND_Y - obs_height - self.np_random.integers(60, 120)

        is_breakable = self.steps > 200 and self.np_random.random() < 0.3
        obs_type = 'breakable' if is_breakable else 'normal'

        new_obs = {'rect': pygame.Rect(self.WIDTH, y_pos, obs_width, obs_height), 'type': obs_type}
        self.obstacles.append(new_obs)

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        reward = 0

        for obs in self.obstacles[:]:
            if player_rect.colliderect(obs['rect']):
                if obs['type'] == 'breakable' and self.is_dashing:
                    self.obstacles.remove(obs)
                    reward += 5
                    self.score += 5
                    self._create_particles(obs['rect'].center, 30, self.COLOR_BREAKABLE, 2, 6)
                    # sfx: break sound
                else:
                    self.obstacles.remove(obs)
                    self.collision_count += 1
                    self._create_particles(player_rect.center, 40, self.COLOR_OBSTACLE, 3, 8)
                    # sfx: hit sound
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)

    def _create_particles(self, pos, count, color, max_radius, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': vel,
                'radius': self.np_random.uniform(1, max_radius),
                'life': self.np_random.integers(10, 20), 'color': color
            })

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_obstacles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_background(self):
        for y in range(self.HEIGHT):
            mix = y / self.HEIGHT
            color = tuple(self.COLOR_BG_TOP[i] * (1 - mix) + self.COLOR_BG_BOTTOM[i] * mix for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_player(self):
        for t in self.dash_trail:
            alpha = int(255 * (t['life'] / 5))
            trail_surf = pygame.Surface(t['rect'].size, pygame.SRCALPHA)
            trail_surf.fill((*self.COLOR_DASH_TRAIL, alpha))
            self.screen.blit(trail_surf, t['rect'].topleft)

        player_width = self.PLAYER_WIDTH * (1.5 if self.is_dashing else 1.0)
        player_x_offset = -self.PLAYER_WIDTH * 0.25 if self.is_dashing else 0
        player_rect = pygame.Rect(self.player_pos.x + player_x_offset, self.player_pos.y - self.PLAYER_HEIGHT, player_width, self.PLAYER_HEIGHT)
        
        glow_color = self.COLOR_DASH_TRAIL if self.is_dashing else self.COLOR_PLAYER
        glow_size = 20 if self.is_dashing else 10
        glow_rect = player_rect.inflate(glow_size, glow_size)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (*glow_color, 50), glow_surf.get_rect())
        self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_obstacles(self):
        for obs in self.obstacles:
            color = self.COLOR_BREAKABLE if obs['type'] == 'breakable' else self.COLOR_OBSTACLE
            darker_color = tuple(c * 0.7 for c in color)
            pygame.draw.rect(self.screen, darker_color, obs['rect'].move(3,3), border_radius=3)
            pygame.draw.rect(self.screen, color, obs['rect'], border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                alpha = int(255 * (p['life'] / 20))
                color_with_alpha = (*p['color'], alpha)
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color_with_alpha, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {int(self.score):05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        collision_text = self.font_main.render(f"HITS: {self.collision_count}/{self.MAX_COLLISIONS}", True, self.COLOR_TEXT)
        self.screen.blit(collision_text, (self.WIDTH - collision_text.get_width() - 10, 10))
        
        if self.dash_cooldown_timer > 0:
            bar_width, bar_height = 50, 8
            fill_width = bar_width * (1 - self.dash_cooldown_timer / self.DASH_COOLDOWN)
            bar_x = self.player_pos.x + self.PLAYER_WIDTH / 2 - bar_width / 2
            bar_y = self.player_pos.y - self.PLAYER_HEIGHT - 15
            pygame.draw.rect(self.screen, (80,80,80), (bar_x, bar_y, bar_width, bar_height), border_radius=2)
            if fill_width > 0:
                pygame.draw.rect(self.screen, self.COLOR_DASH_TRAIL, (bar_x, bar_y, fill_width, bar_height), border_radius=2)

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

if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Runner")
    
    total_reward = 0
    
    while not done:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0
                done = False

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()