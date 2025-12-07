
# Generated: 2025-08-27T20:14:23.905134
# Source Brief: brief_02390.md
# Brief Index: 2390

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    user_guide = (
        "Controls: Press Space to jump and avoid descending hazards."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Descend through a procedurally generated nightmare, jumping over obstacles and dodging lurking horrors to reach the depths below."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- Colors ---
        self.COLOR_BG = (15, 15, 20)
        self.COLOR_PLAYER = (230, 230, 240)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 30)
        self.COLOR_OBSTACLE = (180, 40, 60)
        self.COLOR_ENEMY = (220, 50, 80)
        self.COLOR_TEXT = (200, 30, 30)
        self.COLOR_PARTICLE_DUST = (80, 80, 80)

        # --- Game Constants ---
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -10
        self.PLAYER_WIDTH = 20
        self.PLAYER_HEIGHT = 30
        self.MAX_STEPS = 1000
        self.LEVEL_DEPTH = 8000
        self.SCROLL_SPEED = 3.0
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_complete = False

        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 4]
        self.player_vel_y = 0
        self.on_ground = False

        self.scroll_y = 0
        self.last_obstacle_spawn_y = 0
        self.last_enemy_spawn_y = 0

        self.obstacles = []
        self.enemies = []
        self.particles = []
        
        # Difficulty reset
        self.enemy_speed_mod = 1.0
        self.obstacle_spawn_prob_mod = 1.0
        self.obstacle_spawn_rate = 120
        self.enemy_spawn_rate = 200

        # Parallax background
        self.bg_specks = [
            self._create_speck_layer(70, 0.2, (40, 40, 50)),
            self._create_speck_layer(50, 0.5, (60, 60, 70)),
            self._create_speck_layer(30, 0.8, (80, 80, 90)),
        ]
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # Unused
        space_held = action[1] == 1
        shift_held = action[2] == 1  # Unused
        
        reward = -0.1  # Time penalty
        self.steps += 1

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 50 == 0:
            self.enemy_speed_mod += 0.05
            self.obstacle_spawn_prob_mod += 0.02

        # --- Update Game State ---
        self.scroll_y += self.SCROLL_SPEED
        
        jump_reward = self._update_player(space_held)
        reward += jump_reward

        self._spawn_entities()
        self._update_entities()
        self._update_particles()
        
        collision_reward, pass_reward, near_miss_reward = self._handle_collisions_and_rewards()
        
        if collision_reward < 0:
            self.game_over = True
            reward = collision_reward
        else:
            reward += pass_reward + near_miss_reward
            self.score += pass_reward + near_miss_reward

        # --- Termination Check ---
        if self.scroll_y >= self.LEVEL_DEPTH and not self.game_over:
            self.game_over = True
            self.level_complete = True
            reward += 100
            self.score += 100
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, space_held):
        jump_reward = 0
        
        self.player_vel_y += self.GRAVITY
        self.on_ground = self.player_pos[1] >= self.SCREEN_HEIGHT - self.PLAYER_HEIGHT
        
        if self.on_ground:
            self.player_pos[1] = self.SCREEN_HEIGHT - self.PLAYER_HEIGHT
            if self.player_vel_y > 1:
                # sound: landing_thud.wav
                self._create_particles((self.player_pos[0], self.player_pos[1] + self.PLAYER_HEIGHT), 5, self.COLOR_PARTICLE_DUST, 2, (-1, 1), (-2, 0))
            self.player_vel_y = 0

        if space_held and self.on_ground:
            # sound: jump.wav
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            check_rect = pygame.Rect(self.player_pos[0] - 50, self.player_pos[1] + self.PLAYER_HEIGHT, 100, 150)
            is_necessary = any(check_rect.colliderect(obs['rect']) for obs in self.obstacles) or \
                           any(check_rect.colliderect(self._get_enemy_rect(e)) for e in self.enemies)
            if not is_necessary:
                jump_reward = -0.2

        self.player_pos[1] += self.player_vel_y
        if self.player_pos[1] < 0:
            self.player_pos[1] = 0
            self.player_vel_y = 0
            
        return jump_reward

    def _spawn_entities(self):
        if self.scroll_y - self.last_obstacle_spawn_y > self.obstacle_spawn_rate / self.obstacle_spawn_prob_mod:
            self.last_obstacle_spawn_y = self.scroll_y
            side = 'left' if self.np_random.random() < 0.5 else 'right'
            width = self.np_random.integers(40, 120)
            world_y = self.scroll_y + self.SCREEN_HEIGHT + 50
            x = 0 if side == 'left' else self.SCREEN_WIDTH - width
            self.obstacles.append({'rect': pygame.Rect(x, world_y, width, 20), 'passed': False})

        if self.scroll_y - self.last_enemy_spawn_y > self.enemy_spawn_rate:
            self.last_enemy_spawn_y = self.scroll_y
            self.enemies.append({
                'world_y': self.scroll_y + self.SCREEN_HEIGHT + self.np_random.integers(50, 150),
                'phase': self.np_random.random() * 2 * math.pi,
                'amplitude': self.np_random.integers(100, (self.SCREEN_WIDTH / 2) - 50),
                'freq': self.np_random.uniform(0.02, 0.05),
                'passed': False, 'near_miss_awarded': False
            })

    def _update_entities(self):
        for enemy in self.enemies:
            enemy['phase'] += enemy['freq'] * self.enemy_speed_mod
        self.obstacles = [o for o in self.obstacles if o['rect'].y > self.scroll_y - 50]
        self.enemies = [e for e in self.enemies if e['world_y'] > self.scroll_y - 50]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)

    def _handle_collisions_and_rewards(self):
        pass_reward, near_miss_reward = 0, 0
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_WIDTH / 2, self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        player_world_y = self.scroll_y + self.player_pos[1]

        for obs in self.obstacles:
            screen_rect = obs['rect'].copy()
            screen_rect.y -= self.scroll_y
            if player_rect.colliderect(screen_rect):
                # sound: player_hit.wav
                self._create_particles(player_rect.center, 20, self.COLOR_PLAYER, 4, (-3, 3), (-3, 3))
                return -100, 0, 0
            if not obs['passed'] and obs['rect'].y < player_world_y:
                obs['passed'] = True
                pass_reward += 1

        player_near_miss_rect = player_rect.inflate(60, 60)
        for enemy in self.enemies:
            enemy_screen_rect = self._get_enemy_rect(enemy)
            enemy_screen_rect.y -= self.scroll_y
            if player_rect.colliderect(enemy_screen_rect):
                # sound: enemy_hit.wav
                self._create_particles(player_rect.center, 20, self.COLOR_ENEMY, 4, (-3, 3), (-3, 3))
                return -100, 0, 0
            if not enemy['near_miss_awarded'] and player_near_miss_rect.colliderect(enemy_screen_rect):
                enemy['near_miss_awarded'] = True
                near_miss_reward += 2
                # sound: near_miss_whoosh.wav
            if not enemy['passed'] and enemy['world_y'] < player_world_y:
                enemy['passed'] = True
                pass_reward += 1

        return 0, pass_reward, near_miss_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_entities()
        self._render_particles()
        if not (self.game_over and not self.level_complete):
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_background(self):
        for layer in self.bg_specks:
            scroll_y_layer = self.scroll_y * layer['speed']
            for speck in layer['specks']:
                y = (speck[1] - scroll_y_layer) % self.SCREEN_HEIGHT
                pygame.draw.rect(self.screen, layer['color'], (speck[0], y, 1, 1))

    def _render_entities(self):
        for obs in self.obstacles:
            screen_rect = obs['rect'].copy()
            screen_rect.y -= self.scroll_y
            if -screen_rect.height < screen_rect.y < self.SCREEN_HEIGHT:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
        for enemy in self.enemies:
            screen_y = enemy['world_y'] - self.scroll_y
            if -50 < screen_y < self.SCREEN_HEIGHT + 50:
                rect = self._get_enemy_rect(enemy)
                rect.y -= self.scroll_y
                pygame.gfxdraw.box(self.screen, rect, self.COLOR_ENEMY)
                eye_x = rect.centerx + 5 if self.player_pos[0] > rect.centerx else rect.centerx - 5
                pygame.gfxdraw.filled_circle(self.screen, int(eye_x), int(rect.centery), 2, (255, 255, 255))

    def _render_player(self):
        px, py = self.player_pos
        player_rect = pygame.Rect(px - self.PLAYER_WIDTH / 2, py, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        glow_radius = int(self.PLAYER_WIDTH * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        p1, p2, p3 = (player_rect.left, player_rect.top + 5), (player_rect.right, player_rect.top + 5), (player_rect.centerx, player_rect.top - 10)
        pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            if p['radius'] > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        depth_percent = min(1.0, self.scroll_y / self.LEVEL_DEPTH)
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, (50,50,50), (10, self.SCREEN_HEIGHT - 20, bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, self.SCREEN_HEIGHT - 20, bar_width * depth_percent, 10))
        
        if self.game_over:
            end_text_str = "THE DEPTHS ARE REACHED" if self.level_complete else "CONSUMED BY THE ABYSS"
            end_font = pygame.font.SysFont("monospace", 48, bold=True)
            end_text = end_font.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_speck_layer(self, num_specks, speed, color):
        return {
            'specks': [(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)) for _ in range(num_specks)],
            'speed': speed, 'color': color
        }

    def _create_particles(self, pos, count, color, radius, vel_x_range, vel_y_range):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(*vel_x_range), self.np_random.uniform(*vel_y_range)],
                'life': self.np_random.integers(15, 30),
                'color': color, 'radius': radius * self.np_random.uniform(0.5, 1.2)
            })

    def _get_enemy_rect(self, enemy):
        size = 30
        x = self.SCREEN_WIDTH / 2 + math.sin(enemy['phase']) * enemy['amplitude']
        return pygame.Rect(x - size / 2, enemy['world_y'], size, size)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Nightmare Descent")
    
    terminated = False
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if terminated and event.type == pygame.KEYDOWN:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            action = env.action_space.sample()
            action[0], action[2] = 0, 0
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()