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


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to select a tower plot. SHIFT to cycle tower type. SPACE to build."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Minimalist tower defense. Place towers to defend your base from waves of geometric enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True  # Set to True for typical RL training

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Critical Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Headless Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Visual Style ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_BASE = (0, 150, 136)  # Teal
        self.COLOR_ENEMY = (229, 57, 53)  # Red
        self.COLOR_PROJECTILE = (255, 238, 88)  # Yellow
        self.COLOR_UI_TEXT = (224, 224, 224)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_HEALTH_GREEN = (76, 175, 80)
        self.COLOR_HEALTH_RED = (60, 60, 60)

        # --- Fonts ---
        # Fallback to a default font if 'Consolas' is not available
        try:
            self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 20)


        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.WIN_STEPS = 600
        self.INITIAL_BASE_HEALTH = 100
        self.INITIAL_GOLD = 100
        self.GOLD_PER_STEP = 0.02

        # --- Game Layout ---
        self.path = [
            pygame.Vector2(-20, 150), pygame.Vector2(150, 150),
            pygame.Vector2(150, 300), pygame.Vector2(450, 300),
            pygame.Vector2(450, 100), pygame.Vector2(self.screen_width + 20, 100)
        ]
        self.base_pos = pygame.Vector2(self.screen_width, 100)
        self.tower_placement_spots = [pygame.Vector2(p[0], p[1]) for p in [(80, 225), (225, 225), (375, 225), (525, 225)]]

        # --- Tower Definitions ---
        self.tower_types = {
            0: {"name": "Cannon", "cost": 25, "range": 80, "damage": 10, "fire_rate": 45, "color": (66, 165, 245)},  # Blue
            1: {"name": "Missile", "cost": 60, "range": 120, "damage": 35, "fire_rate": 90, "color": (255, 167, 38)}  # Orange
        }

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.gold = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos_index = 0
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 0
        self.enemy_base_health = 0
        self.enemy_base_speed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.base_health = self.INITIAL_BASE_HEALTH
        self.gold = self.INITIAL_GOLD

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos_index = 0
        self.selected_tower_type = 0

        self.last_space_held = False
        self.last_shift_held = False

        self.enemy_spawn_timer = 30
        self.enemy_spawn_rate = 60  # Initial spawn rate
        self.enemy_base_health = 20
        self.enemy_base_speed = 1.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Cursor movement (no-op for 0, 1, 2)
        if movement == 3:  # Left
            self.cursor_pos_index = max(0, self.cursor_pos_index - 1)
        elif movement == 4:  # Right
            self.cursor_pos_index = min(len(self.tower_placement_spots) - 1, self.cursor_pos_index + 1)

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)

        # Place tower (on press)
        if space_held and not self.last_space_held:
            spot_pos = self.tower_placement_spots[self.cursor_pos_index]
            tower_def = self.tower_types[self.selected_tower_type]
            is_occupied = any(t['pos'] == spot_pos for t in self.towers)

            if not is_occupied and self.gold >= tower_def['cost']:
                self.gold -= tower_def['cost']
                self.towers.append({
                    "pos": spot_pos,
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    **tower_def
                })
                self._create_particles(spot_pos, tower_def['color'], 20, 2.0)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game State ---
        self.steps += 1
        self.gold += self.GOLD_PER_STEP
        reward += self.GOLD_PER_STEP

        # --- Difficulty Scaling ---
        if self.steps > 0:
            if self.steps % 50 == 0: self.enemy_base_health += 2
            if self.steps % 100 == 0: self.enemy_base_speed = min(3.0, self.enemy_base_speed + 0.1)
            if self.steps % 200 == 0: self.enemy_spawn_rate = max(15, self.enemy_spawn_rate - 5)

        # --- Enemy Spawning ---
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self.enemy_spawn_timer = self.enemy_spawn_rate
            self.enemies.append({
                "pos": pygame.Vector2(self.path[0]),  # FIX: Use pygame.Vector2() to copy
                "health": self.enemy_base_health,
                "max_health": self.enemy_base_health,
                "speed": self.enemy_base_speed,
                "path_index": 1,
                "value": 5 + int(self.enemy_base_health / 20)
            })

        # --- Update Towers ---
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = None
                min_dist = tower['range']
                for enemy in self.enemies:
                    dist = tower['pos'].distance_to(enemy['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy

                if target:
                    tower['cooldown'] = tower['fire_rate']
                    self.projectiles.append({
                        "pos": pygame.Vector2(tower['pos']),  # FIX: Use pygame.Vector2() to copy
                        "target": target,
                        "speed": 5,
                        "damage": tower['damage']
                    })

        # --- Update Projectiles ---
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = (proj['target']['pos'] - proj['pos']).normalize()
            proj['pos'] += direction * proj['speed']

            if proj['pos'].distance_to(proj['target']['pos']) < 5:
                proj['target']['health'] -= proj['damage']
                self.projectiles.remove(proj)
                self._create_particles(proj['pos'], self.COLOR_PROJECTILE, 5, 1.5)

        # --- Update Enemies ---
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.gold += enemy['value']
                reward += 1.0  # Reward for killing an enemy
                self.enemies.remove(enemy)
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15, 2.5)
                continue

            if enemy['path_index'] >= len(self.path):
                self.base_health = max(0, self.base_health - 10)
                self.enemies.remove(enemy)
                if self.base_health <= 0:
                    self.game_over = True
                continue

            target_pos = self.path[enemy['path_index']]
            direction = (target_pos - enemy['pos'])
            if direction.length() < enemy['speed']:
                enemy['pos'] = pygame.Vector2(target_pos)  # FIX: Use pygame.Vector2() to copy
                enemy['path_index'] += 1
            else:
                enemy['pos'] += direction.normalize() * enemy['speed']

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Termination and Reward ---
        terminated = False
        if self.base_health <= 0:
            reward = -50.0
            terminated = True
        elif self.steps >= self.WIN_STEPS:
            reward += 50.0
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS

        self.game_over = terminated or truncated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(pos),  # FIX: Use pygame.Vector2() to copy
                "vel": vel,
                "life": self.np_random.integers(10, 26),
                "color": color
            })

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # --- Render Game Elements ---
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [(int(p.x), int(p.y)) for p in self.path], 20)

        # Draw base
        pygame.draw.circle(self.screen, self.COLOR_BASE, (int(self.base_pos.x - 20), int(self.base_pos.y)), 10)

        # Draw tower placement spots and cursor
        for i, spot in enumerate(self.tower_placement_spots):
            is_occupied = any(t['pos'] == spot for t in self.towers)
            color = (50, 60, 70) if is_occupied else (40, 50, 60)
            pygame.gfxdraw.filled_circle(self.screen, int(spot.x), int(spot.y), 20, color)
            pygame.gfxdraw.aacircle(self.screen, int(spot.x), int(spot.y), 20, (60, 70, 80))

            if i == self.cursor_pos_index:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                cursor_color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in self.COLOR_CURSOR)
                pygame.gfxdraw.aacircle(self.screen, int(spot.x), int(spot.y), 23, cursor_color)
                pygame.gfxdraw.aacircle(self.screen, int(spot.x), int(spot.y), 24, cursor_color)

        # Draw towers
        for tower in self.towers:
            p = tower['pos']
            color = tower['color']
            points = [
                (p.x, p.y - 12),
                (p.x - 10, p.y + 6),
                (p.x + 10, p.y + 6)
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], color)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], color)

        # Draw enemies
        for enemy in self.enemies:
            p = enemy['pos']
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y), 8, tuple(min(255, c + 30) for c in self.COLOR_ENEMY))
            # Health bar
            bar_width = 20
            bar_height = 4
            health_pct = enemy['health'] / enemy['max_health']
            health_bar_pos = (int(p.x - bar_width / 2), int(p.y - 18))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (*health_bar_pos, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (*health_bar_pos, int(bar_width * health_pct), bar_height))

        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (int(proj['pos'].x - 2), int(proj['pos'].y - 2), 4, 4))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p['pos'].x - 2), int(p['pos'].y - 2)))

        # --- Render UI Overlay ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Health
        health_text = self.font_main.render(f"♥ {self.base_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 10))

        # Gold
        gold_text = self.font_main.render(f"♦ {int(self.gold)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gold_text, (15, 35))

        # Steps / Wave
        steps_text = self.font_main.render(f"TIME {self.steps}/{self.WIN_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.screen_width - steps_text.get_width() - 15, 10))

        # Selected Tower Info
        tower_def = self.tower_types[self.selected_tower_type]
        name = tower_def['name'].upper()
        cost = tower_def['cost']

        can_afford = self.gold >= cost
        tower_color = self.COLOR_UI_TEXT if can_afford else self.COLOR_ENEMY

        sel_text = self.font_small.render(f"Build: {name} (Cost: {cost})", True, tower_color)
        text_pos = (self.screen_width / 2 - sel_text.get_width() / 2, self.screen_height - 30)
        self.screen.blit(sel_text, text_pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "enemies": len(self.enemies),
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Unset the dummy driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")

    # --- To display the game in a window ---
    pygame.display.set_caption(env.game_description)
    real_screen = pygame.display.set_mode((env.screen_width, env.screen_height))

    obs, info = env.reset(seed=42)
    done = False
    
    # Game loop
    while not done:
        # Human player input
        action = [0, 0, 0]  # no-op, released, released

        # This manual input handling is for demonstration; an agent would use env.action_space.sample()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: action[0] = 3
        if keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Blit the headless surface to the real screen
        # Need to transpose the observation back to pygame's (width, height, channels) format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)  # Limit frame rate for human playability

        if done:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()