import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:35:18.029808
# Source Brief: brief_01090.md
# Brief Index: 1090
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a cosmic field, manage your size to use portals, and fight off hostile entities while avoiding asteroids."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Hold shift to shrink your ship, which is necessary to fit through certain portals. "
        "Press space to attack entities and activate portals."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30
    MAX_STEPS = 5000
    WIN_SCORE = 100

    # Colors
    COLOR_BG = (16, 0, 32)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_ASTEROID = (255, 68, 0)
    COLOR_ENTITY = (255, 0, 255)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_BAR_BG = (50, 50, 80)
    COLOR_BAR_FILL = (100, 100, 255)

    # Player
    PLAYER_BASE_SPEED = 5.0
    PLAYER_BASE_RADIUS = 10
    PLAYER_MIN_SIZE = 0.4
    PLAYER_MAX_SIZE = 1.2
    PLAYER_SHRINK_RATE = 0.05
    PLAYER_GROW_RATE = 0.02
    PLAYER_ATTACK_RANGE_FACTOR = 1.8
    PLAYER_ATTACK_DAMAGE = 10

    # Asteroids
    ASTEROID_COUNT = 15
    ASTEROID_BASE_SPEED = 1.0
    ASTEROID_MIN_RADIUS = 8
    ASTEROID_MAX_RADIUS = 25

    # Portals
    PORTAL_RADIUS = 25
    PORTAL_ACTIVATION_RANGE = 40
    SIZE_TIERS = {
        0: {'range': (0.4, 0.65), 'color': (255, 255, 0)},  # Small, Yellow
        1: {'range': (0.65, 0.9), 'color': (0, 255, 0)},    # Medium, Green
        2: {'range': (0.9, 1.2), 'color': (0, 255, 255)},  # Large, Cyan
    }

    # Cosmic Entities
    ENTITY_BASE_SPAWN_CHANCE = 0.005
    ENTITY_RADIUS = 12
    ENTITY_SPEED = 1.5
    ENTITY_HEALTH = 50
    ENTITY_DEFEAT_SIZE_THRESHOLD = 0.9 # Player size must be >= this to destroy on contact

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        self.player = {}
        self.asteroids = []
        self.portals = []
        self.entities = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_asteroid_speed = self.ASTEROID_BASE_SPEED
        self.current_entity_spawn_chance = self.ENTITY_BASE_SPAWN_CHANCE

        # The reset method is called in the official API, so no need to call it here.
        # self.reset() 
        # self.validate_implementation() # This should not be in init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.current_asteroid_speed = self.ASTEROID_BASE_SPEED
        self.current_entity_spawn_chance = self.ENTITY_BASE_SPAWN_CHANCE

        self.player = {
            'pos': pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2),
            'size': 1.0, # Start at a medium-large size
            'radius': int(self.PLAYER_BASE_RADIUS * 1.0)
        }

        self.asteroids = []
        for _ in range(self.ASTEROID_COUNT):
            self._spawn_asteroid()

        self.portals = []
        self._spawn_portal()

        self.entities = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0.01 # Small reward for surviving
        
        # --- Update Game Logic ---
        self._update_player(movement, space_held, shift_held)
        self._update_asteroids()
        reward += self._update_entities(space_held)
        self._update_particles()
        self._spawn_new_entities()

        # --- Handle Interactions & Rewards ---
        portal_reward = self._handle_portal_activation(space_held)
        reward += portal_reward
        
        collision_penalty, terminated_by_collision = self._handle_collisions()
        reward += collision_penalty
        
        self.steps += 1
        self._update_difficulty()
        
        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated_by_collision or self.score >= self.WIN_SCORE
        
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100.0
            elif terminated_by_collision:
                reward -= 100.0
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_difficulty(self):
        if self.steps > 0:
            if self.steps % 500 == 0:
                self.current_asteroid_speed = min(5.0, self.current_asteroid_speed + 0.05)
            if self.steps % 250 == 0:
                self.current_entity_spawn_chance = min(0.05, self.current_entity_spawn_chance + 0.001)

    def _update_player(self, movement, space_held, shift_held):
        # Size update
        if shift_held:
            self.player['size'] = max(self.PLAYER_MIN_SIZE, self.player['size'] - self.PLAYER_SHRINK_RATE)
        else:
            self.player['size'] = min(self.PLAYER_MAX_SIZE, self.player['size'] + self.PLAYER_GROW_RATE)
        self.player['radius'] = int(self.PLAYER_BASE_RADIUS * self.player['size'])

        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player['pos'] += move_vec * self.PLAYER_BASE_SPEED
            # Thrust particles
            if self.steps % 2 == 0:
                self._create_particles(self.player['pos'], 1, self.COLOR_PLAYER, -move_vec, 1, 2)

        # Screen wrap
        self.player['pos'].x %= self.SCREEN_WIDTH
        self.player['pos'].y %= self.SCREEN_HEIGHT

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel'] * self.current_asteroid_speed
            asteroid['pos'].x %= self.SCREEN_WIDTH
            asteroid['pos'].y %= self.SCREEN_HEIGHT

    def _update_entities(self, space_held):
        reward = 0
        for entity in self.entities[:]:
            direction = (self.player['pos'] - entity['pos']).normalize()
            entity['pos'] += direction * self.ENTITY_SPEED
            
            # Attack logic
            if space_held:
                dist = self.player['pos'].distance_to(entity['pos'])
                attack_range = self.player['radius'] * self.PLAYER_ATTACK_RANGE_FACTOR
                if dist < attack_range:
                    # sfx: player_attack_zap
                    damage = self.PLAYER_ATTACK_DAMAGE * self.player['size']
                    entity['health'] -= damage
                    self._create_particles(entity['pos'], 2, self.COLOR_ENTITY, direction, 2, 3)

            if entity['health'] <= 0:
                # sfx: entity_explode
                self.score += 10
                reward += 10
                self._create_particles(entity['pos'], 30, self.COLOR_ENTITY, None, 1, 5)
                self.entities.remove(entity)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_portal_activation(self, space_held):
        if not space_held:
            return 0

        for portal in self.portals:
            dist = self.player['pos'].distance_to(portal['pos'])
            if dist < self.PORTAL_ACTIVATION_RANGE:
                size_req = self.SIZE_TIERS[portal['tier']]['range']
                if size_req[0] <= self.player['size'] < size_req[1]:
                    # sfx: portal_warp
                    self.score += 5
                    self.player['pos'] = pygame.Vector2(
                        self.np_random.integers(50, self.SCREEN_WIDTH - 50),
                        self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
                    )
                    self._create_particles(portal['pos'], 50, portal['color'], None, 2, 6)
                    self.portals.remove(portal)
                    self._spawn_portal()
                    return 5.0
        return 0

    def _handle_collisions(self):
        # Player vs Asteroids
        for asteroid in self.asteroids:
            dist = self.player['pos'].distance_to(asteroid['pos'])
            if dist < self.player['radius'] + asteroid['radius']:
                # sfx: player_explode
                self._create_particles(self.player['pos'], 40, self.COLOR_PLAYER, None, 2, 7)
                return 0, True

        # Player vs Entities
        for entity in self.entities:
            dist = self.player['pos'].distance_to(entity['pos'])
            if dist < self.player['radius'] + self.ENTITY_RADIUS:
                # sfx: player_hit_damage
                self._create_particles(self.player['pos'], 20, self.COLOR_PLAYER, None, 1, 4)
                return -5.0, True # Terminate on any contact with entity

        return 0, False

    def _spawn_asteroid(self):
        pos = pygame.Vector2(self.np_random.choice([0, self.SCREEN_WIDTH]), self.np_random.uniform(0, self.SCREEN_HEIGHT))
        if self.np_random.choice([True, False]):
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.choice([0, self.SCREEN_HEIGHT]))
        
        vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
        if vel.length() == 0: vel.x = 1
        vel.normalize_ip()
        
        self.asteroids.append({
            'pos': pos, 'vel': vel, 'radius': self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
        })

    def _spawn_portal(self):
        tier = self.np_random.integers(0, len(self.SIZE_TIERS))
        self.portals.append({
            'pos': pygame.Vector2(
                self.np_random.integers(50, self.SCREEN_WIDTH - 50),
                self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            ),
            'tier': tier,
            'color': self.SIZE_TIERS[tier]['color']
        })

    def _spawn_new_entities(self):
        if self.np_random.random() < self.current_entity_spawn_chance:
            pos = pygame.Vector2(self.np_random.choice([0, self.SCREEN_WIDTH]), self.np_random.uniform(0, self.SCREEN_HEIGHT))
            if self.np_random.choice([True, False]):
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.choice([0, self.SCREEN_HEIGHT]))
            self.entities.append({'pos': pos, 'health': self.ENTITY_HEALTH})
            # sfx: entity_spawn

    def _create_particles(self, pos, count, color, direction=None, min_speed=1, max_speed=3):
        for _ in range(count):
            if direction:
                vel = direction.rotate(self.np_random.uniform(-30, 30)) * self.np_random.uniform(min_speed, max_speed)
            else:
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
                if vel.length() > 0:
                    vel.normalize_ip()
                vel *= self.np_random.uniform(min_speed, max_speed)
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_particles()
        self._render_portals()
        self._render_asteroids()
        self._render_entities()
        if not self.game_over:
            self._render_player()

    def _render_player(self):
        pos = (int(self.player['pos'].x), int(self.player['pos'].y))
        radius = self.player['radius']
        
        # Glow effect
        glow_radius = int(radius * 1.8)
        glow_alpha = 60
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Main body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = (int(asteroid['pos'].x), int(asteroid['pos'].y))
            radius = int(asteroid['radius'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

    def _render_portals(self):
        for portal in self.portals:
            pos = (int(portal['pos'].x), int(portal['pos'].y))
            pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
            radius = int(self.PORTAL_RADIUS * (1 + pulse * 0.1))
            color = portal['color']
            
            # Glow
            glow_radius = int(radius * 2.0)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, 40), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))
            
            # Ring
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius - 1, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius - 2, color)

    def _render_entities(self):
        for entity in self.entities:
            pos = (int(entity['pos'].x), int(entity['pos'].y))
            radius = self.ENTITY_RADIUS
            
            # Glow
            glow_radius = int(radius * 2.5)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_ENTITY, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENTITY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENTITY)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            alpha = max(0, min(255, alpha))
            color = (*p['color'], alpha)
            size = int(p['life'] / 10)
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p['pos'].x) - size, int(p['pos'].y) - size))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Player Size Bar
        bar_w, bar_h = 150, 15
        bar_x, bar_y = 10, self.SCREEN_HEIGHT - bar_h - 10
        size_ratio = (self.player['size'] - self.PLAYER_MIN_SIZE) / (self.PLAYER_MAX_SIZE - self.PLAYER_MIN_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_BAR_FILL, (bar_x, bar_y, int(bar_w * size_ratio), bar_h))
        
        # Portal Target
        if self.portals:
            target_portal = self.portals[0]
            target_color = target_portal['color']
            box_size = 30
            box_x = self.SCREEN_WIDTH - box_size - 10
            box_y = self.SCREEN_HEIGHT - box_size - 10
            
            text = self.font_ui.render("TARGET", True, self.COLOR_UI_TEXT)
            self.screen.blit(text, (box_x - text.get_width() - 5, box_y + 5))
            
            pygame.draw.rect(self.screen, target_color, (box_x, box_y, box_size, box_size))
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (box_x, box_y, box_size, box_size), 1)

        if self.game_over:
            msg = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (0, 255, 128) if self.score >= self.WIN_SCORE else (255, 0, 0)
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player.get('size', 0),
            "asteroids": len(self.asteroids),
            "entities": len(self.entities)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It has been modified to use the correct render mode for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cosmic Drifter")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    running = True
    
    while running:
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            
        if terminated or truncated:
            # Allow reset on key press after game over
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                truncated = False

        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.TARGET_FPS)
        
    env.close()