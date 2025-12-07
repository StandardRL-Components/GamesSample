import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:59:46.841588
# Source Brief: brief_00747.md
# Brief Index: 747
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

# Helper function for drawing anti-aliased shapes with a glow effect
def draw_glow_circle(surface, color, center, radius, glow_strength):
    """Draws a circle with a surrounding glow."""
    for i in range(glow_strength, 0, -1):
        alpha = 150 - (i * (150 // glow_strength))
        glow_color = (*color, max(0, alpha))
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius + i), glow_color)
    pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)
    pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your Sanctum from waves of enemies in this strategic tower defense game. "
        "Manage resources, upgrade towers, and use a powerful ability to survive the night."
    )
    user_guide = (
        "Controls: ←→ to shift the enemy spawn focus. Press space to unleash a Sanctum blast. "
        "Hold shift to upgrade towers."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.BOARD_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.BOARD_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) + 10
        
        self.MAX_STEPS = 1000
        self.INITIAL_SANCTUM_HEALTH = 100
        self.INITIAL_RESOURCES = 50
        self.AOE_ABILITY_COOLDOWN = 90  # 3 seconds at 30fps
        self.AOE_ABILITY_RADIUS = 80
        self.AOE_ABILITY_DAMAGE = 50
        self.UPGRADE_COST_BASE = 25

        # --- Colors ---
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_GRID = (30, 20, 60)
        self.COLOR_SANCTUM = (0, 150, 255)
        self.COLOR_TOWER = (0, 200, 200)
        self.COLOR_ENEMY_WEAK = (255, 50, 50)
        self.COLOR_ENEMY_FAST = (255, 120, 0)
        self.COLOR_ENEMY_STRONG = (200, 0, 100)
        self.COLOR_ENEMY_SHIELDED = (180, 180, 180)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_TIME_STREAM = (100, 255, 100)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sanctum_health = 0
        self.resources = 0
        self.towers = []
        self.active_enemies = []
        self.projectiles = []
        self.particles = []
        self.spawn_locations = []
        self.spawn_focus_index = 0
        self.ability_cooldown_timer = 0
        self.upgrade_target_idx = 0
        self.total_damage_taken = 0
        self.total_kills = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sanctum_health = self.INITIAL_SANCTUM_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.total_damage_taken = 0
        self.total_kills = 0

        self.active_enemies.clear()
        self.projectiles.clear()
        self.particles.clear()

        # Initialize Sanctum and Towers
        cx, cy = self.GRID_SIZE // 2, self.GRID_SIZE // 2
        self.sanctum_pos = self._grid_to_pixels(cx, cy)
        self.towers = [
            {'pos': self._grid_to_pixels(cx - 2, cy), 'level': 1, 'cooldown': 0, 'range': 100, 'damage': 10, 'fire_rate': 30},
            {'pos': self._grid_to_pixels(cx + 1, cy), 'level': 1, 'cooldown': 0, 'range': 100, 'damage': 10, 'fire_rate': 30},
            {'pos': self._grid_to_pixels(cx, cy - 2), 'level': 1, 'cooldown': 0, 'range': 100, 'damage': 10, 'fire_rate': 30},
            {'pos': self._grid_to_pixels(cx, cy + 1), 'level': 1, 'cooldown': 0, 'range': 100, 'damage': 10, 'fire_rate': 30},
        ]
        self.upgrade_target_idx = 0
        
        # Initialize Time Stream and Spawn points
        self._generate_spawn_locations()
        self.spawn_focus_index = len(self.spawn_locations) // 2
        self.ability_cooldown_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Pause game if no-op action
        if movement == 0:
            return self._get_observation(), 0, False, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # 1. Handle Player Input
        self._handle_player_input(movement, space_held, shift_held)

        # 2. Update Game Logic
        self._update_cooldowns()
        self._update_spawning()
        damage_taken = self._update_enemies()
        kills = self._update_towers_and_projectiles()
        self._update_particles()
        
        # 3. Calculate Rewards
        self.total_damage_taken += damage_taken
        self.total_kills += kills
        reward -= damage_taken * 0.1
        reward += kills * 1.0
        self.score += kills

        # 4. Check Termination
        terminated = False
        if self.sanctum_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100  # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward += 100  # Win bonus
        
        # Clamp reward to specified range for non-terminal steps
        if not terminated:
            reward = np.clip(reward, -10, 10)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_input(self, movement, space_held, shift_held):
        # Time Stream Manipulation
        if movement == 3:  # Left
            self.spawn_focus_index = (self.spawn_focus_index - 1) % len(self.spawn_locations)
        elif movement == 4:  # Right
            self.spawn_focus_index = (self.spawn_focus_index + 1) % len(self.spawn_locations)

        # AoE Ability
        if space_held and self.ability_cooldown_timer <= 0:
            self.ability_cooldown_timer = self.AOE_ABILITY_COOLDOWN
            # SFX: ZAP_SOUND
            for _ in range(50):
                self.particles.append(self._create_particle(self.sanctum_pos, self.COLOR_TIME_STREAM, 20, 60))
            for enemy in self.active_enemies:
                if math.dist(enemy['pos'], self.sanctum_pos) < self.AOE_ABILITY_RADIUS:
                    enemy['health'] -= self.AOE_ABILITY_DAMAGE
                    # SFX: ENEMY_HIT_SOUND
                    for _ in range(5):
                        self.particles.append(self._create_particle(enemy['pos'], self.COLOR_ENEMY_WEAK, 5, 20, life=15))

        # Upgrade Tower
        if shift_held:
            tower = self.towers[self.upgrade_target_idx]
            cost = self.UPGRADE_COST_BASE * tower['level']
            if self.resources >= cost:
                # SFX: UPGRADE_SOUND
                self.resources -= cost
                tower['level'] += 1
                tower['damage'] *= 1.2
                tower['range'] *= 1.05
                tower['fire_rate'] = max(10, tower['fire_rate'] * 0.95)
                self.upgrade_target_idx = (self.upgrade_target_idx + 1) % len(self.towers)

    def _update_cooldowns(self):
        if self.ability_cooldown_timer > 0:
            self.ability_cooldown_timer -= 1
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1

    def _update_spawning(self):
        spawn_rate = 0.02 + (self.steps // 100) * 0.005
        if self.np_random.random() < spawn_rate:
            # Initial wave guarantee
            if self.steps < 20 and len(self.active_enemies) < 2:
                 enemy_type = 0
            else:
                # Introduce new enemy types over time
                type_pool = [0]
                if self.steps > 200: type_pool.append(1)
                if self.steps > 400: type_pool.append(2)
                if self.steps > 600: type_pool.append(1) # More fast ones
                if self.steps > 800: type_pool.append(3)
                enemy_type = self.np_random.choice(type_pool)

            spawn_pos = self.spawn_locations[self.spawn_focus_index]
            self.active_enemies.append(self._create_enemy(spawn_pos, enemy_type))

    def _update_enemies(self):
        damage_this_step = 0
        for enemy in self.active_enemies[:]:
            # Move towards sanctum
            direction = (self.sanctum_pos[0] - enemy['pos'][0], self.sanctum_pos[1] - enemy['pos'][1])
            dist = math.hypot(*direction)
            if dist < 10: # Reached sanctum
                self.sanctum_health -= enemy['damage']
                damage_this_step += enemy['damage']
                self.active_enemies.remove(enemy)
                # SFX: SANCTUM_HIT_SOUND
                for _ in range(10): self.particles.append(self._create_particle(self.sanctum_pos, self.COLOR_ENEMY_WEAK, 10, 30))
                continue
            
            norm_direction = (direction[0] / dist, direction[1] / dist)
            enemy['pos'] = (enemy['pos'][0] + norm_direction[0] * enemy['speed'], 
                            enemy['pos'][1] + norm_direction[1] * enemy['speed'])
        return damage_this_step

    def _update_towers_and_projectiles(self):
        kills = 0
        # Towers find targets and fire
        for tower in self.towers:
            if tower['cooldown'] <= 0:
                target = None
                min_dist = tower['range']
                for enemy in self.active_enemies:
                    d = math.dist(tower['pos'], enemy['pos'])
                    if d < min_dist:
                        min_dist = d
                        target = enemy
                
                if target:
                    tower['cooldown'] = tower['fire_rate']
                    self.projectiles.append(self._create_projectile(tower['pos'], target, tower['damage']))
                    # SFX: TOWER_SHOOT_SOUND

        # Update projectiles
        for p in self.projectiles[:]:
            if p['target'] not in self.active_enemies:
                self.projectiles.remove(p)
                continue

            direction = (p['target']['pos'][0] - p['pos'][0], p['target']['pos'][1] - p['pos'][1])
            dist = math.hypot(*direction)

            if dist < p['speed']:
                p['target']['health'] -= p['damage']
                self.projectiles.remove(p)
                # SFX: PROJECTILE_HIT_SOUND
                for _ in range(5): self.particles.append(self._create_particle(p['pos'], self.COLOR_PROJECTILE, 3, 15, life=10))
                if p['target']['health'] <= 0:
                    kills += 1
                    self.resources += p['target']['reward']
                    # SFX: ENEMY_DEATH_SOUND
                    for _ in range(20): self.particles.append(self._create_particle(p['target']['pos'], p['target']['color'], 5, 25))
                    self.active_enemies.remove(p['target'])
                continue

            norm_dir = (direction[0] / dist, direction[1] / dist)
            p['pos'] = (p['pos'][0] + norm_dir[0] * p['speed'], p['pos'][1] + norm_dir[1] * p['speed'])
        return kills

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sanctum_health": self.sanctum_health,
            "resources": self.resources,
            "enemies": len(self.active_enemies),
            "total_damage_taken": self.total_damage_taken,
            "total_kills": self.total_kills,
        }

    # --- Rendering Methods ---

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            start_x, end_x = self.BOARD_OFFSET_X, self.BOARD_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE
            start_y, end_y = self.BOARD_OFFSET_Y + i * self.CELL_SIZE, self.BOARD_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))
            start_x, end_x = self.BOARD_OFFSET_X + i * self.CELL_SIZE, self.BOARD_OFFSET_X + i * self.CELL_SIZE
            start_y, end_y = self.BOARD_OFFSET_Y, self.BOARD_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))

        # Draw Sanctum
        draw_glow_circle(self.screen, self.COLOR_SANCTUM, self.sanctum_pos, self.CELL_SIZE * 0.7, 15)

        # Draw Towers
        for i, tower in enumerate(self.towers):
            draw_glow_circle(self.screen, self.COLOR_TOWER, tower['pos'], self.CELL_SIZE * 0.3 * (1 + tower['level'] * 0.1), 5 + tower['level'])
            if i == self.upgrade_target_idx and not self.game_over:
                pygame.gfxdraw.aacircle(self.screen, int(tower['pos'][0]), int(tower['pos'][1]), int(self.CELL_SIZE * 0.5), self.COLOR_GOLD)
        
        # Draw Projectiles
        for p in self.projectiles:
            draw_glow_circle(self.screen, self.COLOR_PROJECTILE, p['pos'], 3, 3)

        # Draw Enemies
        for enemy in self.active_enemies:
            draw_glow_circle(self.screen, enemy['color'], enemy['pos'], enemy['size'], 5)
            # Health bar
            bar_w = self.CELL_SIZE * 0.8
            bar_h = 4
            bar_x = enemy['pos'][0] - bar_w / 2
            bar_y = enemy['pos'][1] - enemy['size'] - 8
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_w * health_pct, bar_h))
        
        # Draw Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            p_color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p_color)

    def _render_ui(self):
        # Time Stream Bar
        bar_y = 15
        bar_h = 25
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, bar_y-5, self.WIDTH, bar_h+10))
        num_spawns_to_show = len(self.spawn_locations)
        for i in range(num_spawns_to_show):
            x = self.WIDTH * (i / num_spawns_to_show)
            w = self.WIDTH / num_spawns_to_show
            if i == self.spawn_focus_index:
                pygame.draw.rect(self.screen, self.COLOR_TIME_STREAM, (x, bar_y, w, bar_h))
            pygame.draw.line(self.screen, self.COLOR_BG, (x, bar_y), (x, bar_y + bar_h))

        # Text Info
        texts = [
            f"TIME: {self.steps}/{self.MAX_STEPS}",
            f"SCORE: {self.score}",
            f"HEALTH: {max(0, int(self.sanctum_health))}",
            f"RESOURCES: {self.resources}"
        ]
        for i, text in enumerate(texts):
            surf = self.font_main.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surf, (10, 50 + i * 22))

        # Ability Cooldown
        ability_text = self.font_main.render("ABILITY", True, self.COLOR_TEXT)
        self.screen.blit(ability_text, (self.WIDTH - 100, 50))
        cd_pct = self.ability_cooldown_timer / self.AOE_ABILITY_COOLDOWN if self.ability_cooldown_timer > 0 else 1.0
        cd_color = self.COLOR_TIME_STREAM if cd_pct >= 1.0 else self.COLOR_GRID
        pygame.draw.rect(self.screen, cd_color, (self.WIDTH - 100, 72, 80 * cd_pct, 10))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH - 100, 72, 80, 10), 1)

        # Game Over Text
        if self.game_over:
            outcome_text = "NIGHT SURVIVED" if self.sanctum_health > 0 else "SANCTUM DESTROYED"
            color = self.COLOR_TIME_STREAM if self.sanctum_health > 0 else self.COLOR_ENEMY_WEAK
            surf = pygame.font.SysFont("Consolas", 48, bold=True).render(outcome_text, True, color)
            text_rect = surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(surf, text_rect)

    # --- Helper Methods ---

    def _grid_to_pixels(self, r, c):
        x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2
        return (x, y)
    
    def _generate_spawn_locations(self):
        locs = []
        # Top and bottom edges
        for i in range(self.GRID_SIZE):
            locs.append(self._grid_to_pixels(-1, i))
            locs.append(self._grid_to_pixels(self.GRID_SIZE, i))
        # Left and right edges
        for i in range(self.GRID_SIZE):
            locs.append(self._grid_to_pixels(i, -1))
            locs.append(self._grid_to_pixels(i, self.GRID_SIZE))
        self.spawn_locations = locs

    def _create_enemy(self, pos, type_id):
        difficulty_mod = 1.0 + (self.steps // 100) * 0.01
        base_stats = [
            {'health': 50, 'speed': 0.8, 'damage': 5, 'size': 8, 'color': self.COLOR_ENEMY_WEAK, 'reward': 10}, # 0: Crawler
            {'health': 30, 'speed': 1.5, 'damage': 3, 'size': 6, 'color': self.COLOR_ENEMY_FAST, 'reward': 12}, # 1: Sprinter
            {'health': 120, 'speed': 0.5, 'damage': 10, 'size': 12, 'color': self.COLOR_ENEMY_STRONG, 'reward': 20}, # 2: Brute
            {'health': 80, 'speed': 0.7, 'damage': 8, 'size': 10, 'color': self.COLOR_ENEMY_SHIELDED, 'reward': 15}, # 3: Shielded
        ][type_id]

        return {
            'pos': pos,
            'health': base_stats['health'] * difficulty_mod,
            'max_health': base_stats['health'] * difficulty_mod,
            'speed': base_stats['speed'],
            'damage': base_stats['damage'] * difficulty_mod,
            'size': base_stats['size'],
            'color': base_stats['color'],
            'reward': base_stats['reward'],
        }

    def _create_projectile(self, pos, target, damage):
        return {'pos': pos, 'target': target, 'damage': damage, 'speed': 5}

    def _create_particle(self, pos, color, radius_range, speed_range, life=30):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, speed_range)
        return {
            'pos': list(pos),
            'vel': [math.cos(angle) * speed / 10, math.sin(angle) * speed / 10],
            'color': color,
            'radius': self.np_random.uniform(1, radius_range/10),
            'life': life,
            'max_life': life
        }

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use Pygame for human interaction
    # NOTE: This will fail if SDL_VIDEODRIVER is "dummy".
    # To run interactively, unset the environment variable before running this script.
    # For example: SDL_VIDEODRIVER=x11 python your_script.py
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Nightmare Sanctum")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Human input mapping
        keys = pygame.key.get_pressed()
        movement = 0 # no-op/pause
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # In human play, we always want the game to advance unless paused
        # So we default to a non-zero movement action if nothing is pressed
        if movement == 0 and not space_held and not shift_held:
            action = [0, 0, 0] # Explicit pause
        else:
            # If any other key is pressed, we need a non-zero movement to unpause
            # We'll use a dummy non-moving action if only space/shift are held.
            # The spec says action[0]==0 is pause, so we must send something else.
            # Let's just pass the current movement state.
            action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}")
            print(f"Final Info: {info}")
            # In a real game, you might wait for a keypress to reset
            # obs, info = env.reset()
            # total_reward = 0
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()