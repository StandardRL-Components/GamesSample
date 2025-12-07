import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Chord Defense: A Tower Defense game where players place 'Chord Towers' to
    defeat waves of enemies. Combining different chord attacks in sequence
    triggers powerful chain reactions. The goal is to survive 20 waves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Chord Defense: A Tower Defense game where players place 'Chord Towers' to "
        "defeat waves of enemies. Combining different chord attacks in sequence "
        "triggers powerful chain reactions."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to place a tower and shift to "
        "cycle through tower types."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 20000 # Increased to allow for a full game

    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (20, 40, 60)
    COLOR_PATH = (30, 60, 90)
    COLOR_BASE = (200, 50, 50)
    COLOR_ENEMY = (220, 220, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
    CHORD_TYPES = {
        "Major": {"color": (255, 80, 80), "symbol": "M", "cost": 100, "range": 80, "fire_rate": 45, "damage": 20},
        "Minor": {"color": (80, 120, 255), "symbol": "m", "cost": 125, "range": 100, "fire_rate": 60, "damage": 15},
        "Diminished": {"color": (180, 80, 255), "symbol": "d", "cost": 150, "range": 70, "fire_rate": 30, "damage": 10},
        "Augmented": {"color": (80, 255, 150), "symbol": "A", "cost": 200, "range": 120, "fire_rate": 90, "damage": 40},
    }
    CHORD_SEQUENCE = list(CHORD_TYPES.keys())
    
    # Chain Reactions: (Chord 1, Chord 2) -> effect
    CHAIN_REACTIONS = {
        ("Major", "Minor"): {"radius": 70, "damage": 50},
        ("Minor", "Diminished"): {"radius": 50, "damage": 80},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_tower = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.resources = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.current_chord_idx = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        self.path = self._define_path()
        self.base_pos = self.path[-1]

    def _define_path(self):
        return [
            (-20, 50), (100, 50), (100, 200), (300, 200),
            (300, 120), (500, 120), (500, 300), (self.SCREEN_WIDTH + 20, 300)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.resources = 250
        self.wave_number = 0
        self.wave_timer = self.FPS * 5 # 5 seconds to first wave
        self.enemies_to_spawn = 0
        self.spawn_timer = 0

        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.current_chord_idx = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle player input
        reward += self._handle_input(movement, space_held, shift_held)

        # Update game logic
        wave_cleared_reward = self._update_waves()
        reward += wave_cleared_reward
        
        self._update_enemies()
        hit_reward, chain_reward = self._update_towers()
        reward += hit_reward + chain_reward
        
        self._update_projectiles()
        self._update_particles()
        
        # Continuous penalty for enemies on screen
        reward -= 0.001 * len(self.enemies)

        # Check for termination conditions
        if any(enemy['reached_base'] for enemy in self.enemies):
            self.game_over = True
            self.victory = False
            reward -= 100
        elif self.wave_number > 20 and not self.enemies:
            self.game_over = True
            self.victory = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward -= 20 # Penalty for timeout
        
        terminated = self.game_over

        # Update button press states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        
        # Move cursor
        cursor_speed = 8
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Cycle chord type on SHIFT press
        if shift_held and not self.prev_shift_held:
            self.current_chord_idx = (self.current_chord_idx + 1) % len(self.CHORD_SEQUENCE)
            # sfx: ui_blip

        # Place tower on SPACE press
        if space_held and not self.prev_space_held:
            chord_name = self.CHORD_SEQUENCE[self.current_chord_idx]
            chord_data = self.CHORD_TYPES[chord_name]
            
            if self.resources >= chord_data['cost'] and self._is_valid_placement(self.cursor_pos):
                self.resources -= chord_data['cost']
                self.towers.append({
                    "pos": list(self.cursor_pos),
                    "type": chord_name,
                    "cooldown": 0,
                    "fire_rate": chord_data['fire_rate']
                })
                # sfx: place_tower
                self._create_particles(self.cursor_pos, chord_data['color'], 20, 3)
                reward += 0.5 # Small reward for placing a tower
        
        return reward

    def _is_valid_placement(self, pos):
        # Cannot place on path
        path_clearance = 25
        for i in range(len(self.path) - 1):
            p1 = np.array(self.path[i])
            p2 = np.array(self.path[i+1])
            p3 = np.array(pos)
            
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1) if np.linalg.norm(p2-p1) != 0 else np.linalg.norm(p1-p3)
            
            # Check if point is within the segment bounds
            dot_product = (p3 - p1).dot(p2 - p1)
            if 0 <= dot_product <= (p2 - p1).dot(p2 - p1):
                if d < path_clearance:
                    return False
        
        # Cannot place too close to another tower
        tower_clearance = 40
        for tower in self.towers:
            if math.dist(pos, tower['pos']) < tower_clearance:
                return False
                
        return True

    def _update_waves(self):
        if self.enemies_to_spawn == 0 and not self.enemies:
            if self.wave_timer > 0:
                self.wave_timer -= 1
            else:
                self.wave_number += 1
                if self.wave_number > 20: return 0 # Game won
                
                # sfx: wave_start
                self.enemies_to_spawn = 3 + (self.wave_number - 1)
                self.spawn_timer = 0
                return 5.0 # Wave clear reward
        
        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn -= 1
                self.spawn_timer = self.FPS * 0.5 # Spawn every 0.5 seconds
        return 0

    def _spawn_enemy(self):
        health = 100 * (1.1 ** (self.wave_number - 1))
        speed = 1.0 * (1.02 ** (self.wave_number - 1))
        self.enemies.append({
            "pos": list(self.path[0]),
            "health": health,
            "max_health": health,
            "speed": speed,
            "path_idx": 1,
            "dist_on_segment": 0,
            "reached_base": False,
            "mark": None,
            "mark_timer": 0,
        })

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy['reached_base']: continue

            if enemy['mark_timer'] > 0:
                enemy['mark_timer'] -= 1
            else:
                enemy['mark'] = None

            target_waypoint = self.path[enemy['path_idx']]
            direction = np.array(target_waypoint) - np.array(enemy['pos'])
            distance = np.linalg.norm(direction)

            if distance < enemy['speed']:
                enemy['path_idx'] += 1
                if enemy['path_idx'] >= len(self.path):
                    enemy['reached_base'] = True
                    # sfx: lose_life
                    continue
            else:
                move_vec = (direction / distance) * enemy['speed']
                enemy['pos'][0] += move_vec[0]
                enemy['pos'][1] += move_vec[1]
                
    def _update_towers(self):
        hit_reward = 0
        chain_reward = 0
        
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            chord_data = self.CHORD_TYPES[tower['type']]
            target = None
            min_dist = chord_data['range']

            for enemy in self.enemies:
                dist = math.dist(tower['pos'], enemy['pos'])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower['fire_rate']
                self.projectiles.append({
                    "start_pos": list(tower['pos']),
                    "pos": list(tower['pos']),
                    "target": target,
                    "type": tower['type'],
                    "speed": 8,
                })
                # sfx: shoot
                
        # Handle projectile hits (moved from a separate function for easier reward passing)
        for proj in self.projectiles[:]:
            if not proj['target'] in self.enemies:
                if proj in self.projectiles:
                    self.projectiles.remove(proj)
                continue

            dist_to_target = math.dist(proj['pos'], proj['target']['pos'])
            if dist_to_target < 5:
                # sfx: hit_enemy
                hit_reward += 0.1
                
                # Apply damage
                damage = self.CHORD_TYPES[proj['type']]['damage']
                proj['target']['health'] -= damage
                
                # Create hit particles
                self._create_particles(proj['pos'], self.CHORD_TYPES[proj['type']]['color'], 10, 2)
                
                # Check for chain reactions
                if proj['target']['mark'] and proj['target']['mark_timer'] > 0:
                    progression = (proj['target']['mark'], proj['type'])
                    if progression in self.CHAIN_REACTIONS:
                        # sfx: chain_reaction_explosion
                        chain_reward += 1.0
                        reaction_data = self.CHAIN_REACTIONS[progression]
                        self._trigger_chain_reaction(proj['pos'], reaction_data)
                        proj['target']['mark'] = None # Consume mark
                    else: # Apply new mark
                        proj['target']['mark'] = proj['type']
                        proj['target']['mark_timer'] = self.FPS * 2 # 2 seconds
                else: # No mark, apply new one
                    proj['target']['mark'] = proj['type']
                    proj['target']['mark_timer'] = self.FPS * 2

                if proj in self.projectiles:
                    self.projectiles.remove(proj)

        # Cleanup dead enemies
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                # sfx: enemy_die
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 30, 4)
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
                reward_for_kill = 0.2 * (1.05 ** (self.wave_number - 1))
                hit_reward += reward_for_kill
                
        return hit_reward, chain_reward

    def _trigger_chain_reaction(self, pos, reaction_data):
        self._create_particles(pos, (255, 255, 255), 50, 8, is_explosion=True)
        for enemy in self.enemies:
            if math.dist(pos, enemy['pos']) < reaction_data['radius']:
                enemy['health'] -= reaction_data['damage']
                # Create secondary hit particles
                self._create_particles(enemy['pos'], (200, 200, 200), 5, 1)

    def _update_projectiles(self):
        for proj in self.projectiles:
            if not proj['target'] in self.enemies:
                continue
            target_pos = proj['target']['pos']
            direction = np.array(target_pos) - np.array(proj['pos'])
            distance = np.linalg.norm(direction)
            if distance > 0:
                move_vec = (direction / distance) * proj['speed']
                proj['pos'][0] += move_vec[0]
                proj['pos'][1] += move_vec[1]
                
    def _create_particles(self, pos, color, count, speed_scale, is_explosion=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_scale
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(10, 20) if not is_explosion else random.randint(20, 40)
            self.particles.append({
                "pos": list(pos), "vel": vel, "lifespan": lifespan, 
                "max_life": lifespan, "color": color, "size": random.uniform(1, 3)
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_path()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_path(self):
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 20)
        
        # Base
        base_line_start = (self.base_pos[0] - 15, self.base_pos[1])
        base_line_end = (self.base_pos[0] + 15, self.base_pos[1])
        pygame.draw.line(self.screen, self.COLOR_BASE, base_line_start, base_line_end, 5)

    def _render_towers(self):
        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            chord_data = self.CHORD_TYPES[tower['type']]
            color = chord_data['color']
            
            # Glow
            glow_radius = 22
            glow_color = (color[0] // 4, color[1] // 4, color[2] // 4)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)
            
            # Tower body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, tuple(c//2 for c in color))
            
            # Symbol
            symbol_surf = self.font_tower.render(chord_data['symbol'], True, self.COLOR_BG)
            self.screen.blit(symbol_surf, (pos[0] - symbol_surf.get_width()//2, pos[1] - symbol_surf.get_height()//2))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            
            # Health bar
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            health_bar_width = 20
            health_bar_height = 4
            health_bar_pos = (pos[0] - health_bar_width // 2, pos[1] - 20)
            
            pygame.draw.rect(self.screen, (80, 0, 0), (*health_bar_pos, health_bar_width, health_bar_height))
            pygame.draw.rect(self.screen, (0, 180, 0), (*health_bar_pos, int(health_bar_width * health_pct), health_bar_height))

            # Enemy body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, tuple(c//2 for c in self.COLOR_ENEMY))
            
            # Mark indicator
            if enemy['mark'] and enemy['mark_timer'] > 0:
                mark_color = self.CHORD_TYPES[enemy['mark']]['color']
                alpha = int(255 * (enemy['mark_timer'] / (self.FPS * 2)))
                mark_color_alpha = (*mark_color, alpha)
                
                s = pygame.Surface((18, 18), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, 9, 9, 9, mark_color_alpha)
                self.screen.blit(s, (pos[0] - 9, pos[1] - 9))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            color = self.CHORD_TYPES[proj['type']]['color']
            
            # Trail
            start_pos = (int(proj['start_pos'][0]), int(proj['start_pos'][1]))
            dx, dy = pos[0] - start_pos[0], pos[1] - start_pos[1]
            dist = math.hypot(dx, dy)
            if dist > 0:
                trail_len = min(dist, 20)
                trail_start = (pos[0] - (dx/dist)*trail_len, pos[1] - (dy/dist)*trail_len)
                pygame.draw.line(self.screen, color, trail_start, pos, 2)

            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, tuple(c//4 for c in color))
            # Core
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color = (*p['color'], alpha)
            size = p['size']
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (p['pos'][0] - size, p['pos'][1] - size))

    def _render_cursor(self):
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        chord_name = self.CHORD_SEQUENCE[self.current_chord_idx]
        chord_data = self.CHORD_TYPES[chord_name]
        
        is_valid = self._is_valid_placement(self.cursor_pos)
        can_afford = self.resources >= chord_data['cost']
        
        if not is_valid: color = (255, 0, 0, 100)
        elif not can_afford: color = (255, 255, 0, 100)
        else: color = (255, 255, 255, 100)
        
        # Range indicator
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, pos[0], pos[1], chord_data['range'], color)
        self.screen.blit(s, (0,0))
        
        # Cursor crosshair
        pygame.draw.line(self.screen, color[:3], (pos[0]-10, pos[1]), (pos[0]+10, pos[1]))
        pygame.draw.line(self.screen, color[:3], (pos[0], pos[1]-10), (pos[0], pos[1]+10))

    def _render_ui(self):
        def draw_text(text, pos, font, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            shadow = font.render(text, True, shadow_color)
            self.screen.blit(shadow, (pos[0]+1, pos[1]+1))
            main = font.render(text, True, color)
            self.screen.blit(main, pos)

        draw_text(f"SCORE: {int(self.score)}", (10, 10), self.font_ui)
        draw_text(f"RESOURCES: {self.resources}", (10, 30), self.font_ui)
        
        wave_text = f"WAVE: {self.wave_number}/20"
        if self.wave_timer > 0 and self.wave_number < 20:
            wave_text += f" (in {self.wave_timer/self.FPS:.1f}s)"
        draw_text(wave_text, (self.SCREEN_WIDTH - 250, 10), self.font_ui)

        # Current selection
        chord_name = self.CHORD_SEQUENCE[self.current_chord_idx]
        chord_data = self.CHORD_TYPES[chord_name]
        draw_text(f"SELECT: {chord_name} ({chord_data['cost']})", (self.SCREEN_WIDTH - 250, 30), self.font_ui, color=chord_data['color'])

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0,0))
        
        text = "VICTORY!" if self.victory else "GAME OVER"
        color = (100, 255, 100) if self.victory else (255, 100, 100)
        
        text_surf = self.font_game_over.render(text, True, color)
        pos = (self.SCREEN_WIDTH//2 - text_surf.get_width()//2, self.SCREEN_HEIGHT//2 - text_surf.get_height()//2)
        self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "towers": len(self.towers),
            "enemies": len(self.enemies),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It requires a graphical display, which is different from the headless 'rgb_array' mode
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chord Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to restart.")
        
        clock.tick(GameEnv.FPS)
        
    env.close()