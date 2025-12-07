import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:55:37.797267
# Source Brief: brief_00703.md
# Brief Index: 703
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player leads a cloned jungle animal,
    manipulating magnetic fields and time to defeat enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a cloned jungle animal with unique powers. Defeat enemies using magnetic fields, "
        "time dilation, and powerful dashes to survive the onslaught."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Releasing movement keys switches animal. "
        "Press space for abilities and hold shift to slow time."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2500
    BOSS_SPAWN_SCORE = 30

    # Colors
    COLOR_BG = (20, 40, 30)
    COLOR_PLAYER = (255, 180, 0)
    COLOR_PLAYER_GLOW = (255, 180, 0, 50)
    COLOR_ENEMY = (200, 40, 80)
    COLOR_ENEMY_GLOW = (200, 40, 80, 50)
    COLOR_BOSS = (150, 50, 255)
    COLOR_BOSS_GLOW = (150, 50, 255, 50)
    COLOR_RESOURCE = (50, 200, 255)
    COLOR_RESOURCE_GLOW = (50, 200, 255, 60)
    COLOR_MAGNETIC_PULSE = (100, 150, 255)
    COLOR_STUN_WAVE = (220, 220, 100)
    COLOR_TIME_FX = (120, 80, 200, 30)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_HEALTH_BAR = (40, 200, 60)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)

    # Animal types
    ANIMAL_MONKEY = 0
    ANIMAL_JAGUAR = 1
    ANIMAL_PARROT = 2
    ANIMAL_NAMES = ["Monkey (Push)", "Jaguar (Dash)", "Parrot (Stun)"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_animal = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State Initialization ---
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_max_health = 100
        self.player_radius = 12
        self.player_speed = 200
        self.selected_animal = None
        self.ability_cooldown = 0
        self.dash_state = {"active": False, "duration": 0, "vel": pygame.Vector2(0,0)}

        self.enemies = []
        self.boss = None
        self.boss_spawned = False
        self.resources = []
        self.particles = []
        self.active_effects = []

        self.time_dilation_factor = 1.0
        self.last_movement_action = 0

        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.boss_spawned = False
        self.last_movement_action = 1 # Default to 'up'

        # Player
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.player_max_health
        self.selected_animal = self.ANIMAL_MONKEY
        self.ability_cooldown = 0
        self.dash_state = {"active": False, "duration": 0, "vel": pygame.Vector2(0,0)}

        # Entities
        self.enemies = []
        self.boss = None
        self.resources = []
        self.particles = []
        self.active_effects = []
        
        # Level Generation
        for _ in range(5):
            self._spawn_enemy()
        for _ in range(8):
            self._spawn_resource()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Time Management ---
        # Use a fixed delta time for consistent physics, adjusted by time dilation
        dt = (1 / self.FPS) * self.time_dilation_factor
        
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle time dilation
        self.time_dilation_factor = 0.5 if shift_held else 1.0
        # Placeholder for sound effect
        # if shift_held and self.time_dilation_factor != 0.5: play_sound('time_slow_start')
        # if not shift_held and self.time_dilation_factor != 1.0: play_sound('time_slow_end')

        # Handle animal switching (on press of 'no movement')
        if movement == 0 and self.last_movement_action != 0:
            self.selected_animal = (self.selected_animal + 1) % 3
            # Placeholder for sound effect
            # play_sound('switch_animal')
        self.last_movement_action = movement

        # Handle movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_vel = move_vec
        else:
            self.player_vel = pygame.Vector2(0, 0)
            
        # Handle abilities
        self.ability_cooldown = max(0, self.ability_cooldown - dt)
        if space_held and self.ability_cooldown == 0:
            self._activate_ability()

        # --- Update Game Logic ---
        self._update_player(dt)
        self._update_enemies(dt)
        if self.boss:
            self._update_boss(dt)
        self._update_particles(dt)
        self._update_effects(dt)

        # --- Collision & Interaction ---
        reward += self._handle_collisions()

        # --- Game State Progression ---
        if self.score >= self.BOSS_SPAWN_SCORE and not self.boss_spawned:
            self._spawn_boss()
            self.boss_spawned = True
            # Placeholder for sound effect
            # play_sound('boss_spawn')
        
        if len(self.enemies) < 3 and not self.boss_spawned:
            self._spawn_enemy()

        # --- Termination Conditions ---
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 50 # Large penalty for dying
        elif self.boss and self.boss['health'] <= 0:
            terminated = True
            self.game_over = True
            self.score += 100 # Add final boss points to score
            reward += 100 # Large reward for winning
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _activate_ability(self):
        if self.selected_animal == self.ANIMAL_MONKEY:
            self.ability_cooldown = 1.5
            self.active_effects.append({
                "type": "magnetic_pulse", "pos": self.player_pos.copy(),
                "radius": 0, "max_radius": 100, "duration": 0.3, "life": 0
            })
            # Placeholder for sound effect
            # play_sound('magnetic_push')
        elif self.selected_animal == self.ANIMAL_JAGUAR:
            if not self.dash_state["active"]:
                self.ability_cooldown = 2.0
                self.dash_state["active"] = True
                self.dash_state["duration"] = 0.2
                dash_dir = self.player_vel if self.player_vel.length() > 0 else pygame.Vector2(0, -1)
                self.dash_state["vel"] = dash_dir.normalize() * self.player_speed * 4
                # Placeholder for sound effect
                # play_sound('jaguar_dash')
        elif self.selected_animal == self.ANIMAL_PARROT:
            self.ability_cooldown = 3.0
            self.active_effects.append({
                "type": "stun_wave", "pos": self.player_pos.copy(),
                "radius": 0, "max_radius": 120, "duration": 0.25, "life": 0
            })
            # Placeholder for sound effect
            # play_sound('parrot_stun')

    def _update_player(self, dt):
        if self.dash_state["active"]:
            self.player_pos += self.dash_state["vel"] * dt
            self.dash_state["duration"] -= dt
            if self.dash_state["duration"] <= 0:
                self.dash_state["active"] = False
        else:
            self.player_pos += self.player_vel * self.player_speed * dt

        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.player_radius, self.SCREEN_WIDTH - self.player_radius)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_radius, self.SCREEN_HEIGHT - self.player_radius)
    
    def _update_enemies(self, dt):
        for enemy in self.enemies:
            if enemy["stunned_timer"] > 0:
                enemy["stunned_timer"] -= dt
                continue
            
            dist_to_player = self.player_pos.distance_to(enemy['pos'])
            
            if dist_to_player < enemy['aggro_radius']: # Attack state
                direction = (self.player_pos - enemy['pos']).normalize()
                enemy['pos'] += direction * enemy['speed'] * dt
            else: # Patrol state
                target = enemy['path'][enemy['path_index']]
                direction = (target - enemy['pos'])
                if direction.length() < 10:
                    enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
                else:
                    direction.normalize_ip()
                    enemy['pos'] += direction * (enemy['speed'] * 0.6) * dt
            
            enemy['pos'].x = np.clip(enemy['pos'].x, enemy['radius'], self.SCREEN_WIDTH - enemy['radius'])
            enemy['pos'].y = np.clip(enemy['pos'].y, enemy['radius'], self.SCREEN_HEIGHT - enemy['radius'])

    def _update_boss(self, dt):
        if self.boss["stunned_timer"] > 0:
            self.boss["stunned_timer"] -= dt
            return
        
        direction = (self.player_pos - self.boss['pos']).normalize()
        self.boss['pos'] += direction * self.boss['speed'] * dt
        
        self.boss['pos'].x = np.clip(self.boss['pos'].x, self.boss['radius'], self.SCREEN_WIDTH - self.boss['radius'])
        self.boss['pos'].y = np.clip(self.boss['pos'].y, self.boss['radius'], self.SCREEN_HEIGHT - self.boss['radius'])

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p['pos'] += p['vel'] * dt
            p['life'] -= dt
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_effects(self, dt):
        for effect in self.active_effects[:]:
            effect['life'] += dt
            effect['radius'] = (effect['life'] / effect['duration']) * effect['max_radius']
            if effect['life'] >= effect['duration']:
                self.active_effects.remove(effect)

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Resources
        for res in self.resources[:]:
            if self.player_pos.distance_to(res['pos']) < self.player_radius + res['radius']:
                self.resources.remove(res)
                self.score += 1
                reward += 1.0
                self._create_particles(res['pos'], self.COLOR_RESOURCE, 15)
                # Placeholder for sound effect
                # play_sound('collect_resource')

        # Player vs Enemies (if not dashing)
        if not self.dash_state["active"]:
            entities_to_check = self.enemies[:]
            if self.boss: entities_to_check.append(self.boss)
            
            for enemy in entities_to_check:
                if self.player_pos.distance_to(enemy['pos']) < self.player_radius + enemy['radius']:
                    self.player_health = max(0, self.player_health - enemy['damage'])
                    reward -= 0.1
                    self._create_particles(self.player_pos, self.COLOR_ENEMY, 20, intensity=1.5)
                    # Placeholder for sound effect
                    # play_sound('player_hurt')
        
        # Effects vs Enemies
        for effect in self.active_effects:
            entities_to_check = self.enemies[:]
            if self.boss: entities_to_check.append(self.boss)

            for enemy in entities_to_check:
                dist = effect['pos'].distance_to(enemy['pos'])
                if dist < effect['radius'] + enemy['radius']:
                    if effect['type'] == 'magnetic_pulse':
                        direction = (enemy['pos'] - effect['pos']).normalize()
                        enemy['pos'] += direction * 250 * (1/self.FPS) # Immediate push
                    elif effect['type'] == 'stun_wave':
                        enemy['stunned_timer'] = 2.0 # Stun for 2 seconds

        # Dashing Player vs Enemies
        if self.dash_state["active"]:
            entities_to_check = self.enemies[:]
            if self.boss: entities_to_check.append(self.boss)
            
            for enemy in entities_to_check:
                if self.player_pos.distance_to(enemy['pos']) < self.player_radius + enemy['radius']:
                    enemy['health'] -= 35 # Dash damage
                    if enemy['health'] <= 0:
                        reward += self._kill_entity(enemy)
        
        return reward

    def _kill_entity(self, entity):
        # Placeholder for sound effect
        # play_sound('enemy_die')
        self._create_particles(entity['pos'], entity['color'], 30, intensity=2.0)
        
        if 'is_boss' in entity and entity['is_boss']:
            self.boss = None
            self.score += 50
            return 50.0 # Boss kill reward
        else:
            if entity in self.enemies:
                self.enemies.remove(entity)
            self.score += 5
            return 10.0 # Normal enemy kill reward

    def _spawn_enemy(self):
        side = random.randint(0, 3)
        if side == 0: pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), -20)
        elif side == 1: pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
        elif side == 2: pos = pygame.Vector2(-20, random.uniform(0, self.SCREEN_HEIGHT))
        else: pos = pygame.Vector2(self.SCREEN_WIDTH + 20, random.uniform(0, self.SCREEN_HEIGHT))
        
        path = [pygame.Vector2(random.uniform(50, self.SCREEN_WIDTH - 50), random.uniform(50, self.SCREEN_HEIGHT - 50)) for _ in range(2)]
        
        self.enemies.append({
            'pos': pos, 'radius': 10, 'speed': random.uniform(60, 90),
            'health': 100, 'max_health': 100, 'damage': 10,
            'path': path, 'path_index': 0, 'aggro_radius': 150,
            'color': self.COLOR_ENEMY, 'glow_color': self.COLOR_ENEMY_GLOW,
            'stunned_timer': 0
        })

    def _spawn_boss(self):
        self.boss = {
            'pos': pygame.Vector2(self.SCREEN_WIDTH / 2, -50), 'radius': 30, 
            'speed': 70, 'health': 1000, 'max_health': 1000, 'damage': 25,
            'color': self.COLOR_BOSS, 'glow_color': self.COLOR_BOSS_GLOW,
            'stunned_timer': 0, 'is_boss': True
        }

    def _spawn_resource(self):
        self.resources.append({
            'pos': pygame.Vector2(random.uniform(20, self.SCREEN_WIDTH - 20), random.uniform(20, self.SCREEN_HEIGHT - 20)),
            'radius': 6
        })

    def _create_particles(self, pos, color, count, intensity=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150) * intensity
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': random.uniform(0.3, 0.8),
                'color': color
            })

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health}

    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        if self.time_dilation_factor < 1.0:
            time_fx_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            time_fx_surface.fill(self.COLOR_TIME_FX)
            self.screen.blit(time_fx_surface, (0, 0))

        # --- Effects ---
        for effect in self.active_effects:
            color = self.COLOR_MAGNETIC_PULSE if effect['type'] == 'magnetic_pulse' else self.COLOR_STUN_WAVE
            alpha = int(150 * (1 - (effect['life'] / effect['duration'])))
            pygame.gfxdraw.aacircle(self.screen, int(effect['pos'].x), int(effect['pos'].y), int(effect['radius']), (*color, alpha))

        # --- Resources ---
        for res in self.resources:
            glow_radius = int(res['radius'] * (2.5 + math.sin(pygame.time.get_ticks() / 200)))
            pygame.gfxdraw.filled_circle(self.screen, int(res['pos'].x), int(res['pos'].y), glow_radius, self.COLOR_RESOURCE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(res['pos'].x), int(res['pos'].y), res['radius'], self.COLOR_RESOURCE)
            pygame.gfxdraw.aacircle(self.screen, int(res['pos'].x), int(res['pos'].y), res['radius'], self.COLOR_RESOURCE)

        # --- Entities (Enemies and Boss) ---
        entities_to_draw = self.enemies[:]
        if self.boss: entities_to_draw.append(self.boss)
        for enemy in entities_to_draw:
            # Glow
            glow_radius = int(enemy['radius'] * 1.8)
            pygame.gfxdraw.filled_circle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), glow_radius, enemy['glow_color'])
            # Body
            pygame.gfxdraw.filled_circle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), enemy['radius'], enemy['color'])
            pygame.gfxdraw.aacircle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), enemy['radius'], (0,0,0))
            # Health bar
            self._render_health_bar(enemy['pos'], enemy['radius'], enemy['health'], enemy['max_health'])
            # Stunned effect
            if enemy['stunned_timer'] > 0:
                self._render_stun_effect(enemy['pos'], enemy['radius'])

        # --- Player ---
        player_color = self.COLOR_PLAYER
        if self.dash_state["active"]:
            # Dash trail effect
            for i in range(5):
                trail_pos = self.player_pos - self.dash_state["vel"] * (i * 0.01)
                alpha = 150 - i * 30
                pygame.gfxdraw.filled_circle(self.screen, int(trail_pos.x), int(trail_pos.y), self.player_radius, (*player_color, alpha))
        
        # Glow
        glow_radius = int(self.player_radius * 2.0)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), glow_radius, self.COLOR_PLAYER_GLOW)
        # Body
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.player_radius, player_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.player_radius, (0,0,0))
        # Health bar
        self._render_health_bar(self.player_pos, self.player_radius, self.player_health, self.player_max_health)

        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / 0.8))
            color = (*p['color'], alpha)
            size = int(p['life'] * 5 + 2)
            pygame.draw.rect(self.screen, color, (int(p['pos'].x), int(p['pos'].y), size, size))

        # --- UI ---
        self._render_ui()

    def _render_health_bar(self, pos, radius, current, maximum):
        bar_width = radius * 2
        bar_height = 5
        y_offset = radius + 5
        bg_rect = pygame.Rect(pos.x - bar_width/2, pos.y - y_offset, bar_width, bar_height)
        
        health_ratio = np.clip(current / maximum, 0, 1)
        fg_width = int(bar_width * health_ratio)
        fg_rect = pygame.Rect(pos.x - bar_width/2, pos.y - y_offset, fg_width, bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect)
        if fg_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fg_rect)

    def _render_stun_effect(self, pos, radius):
        num_stars = 3
        y_offset = radius + 15
        for i in range(num_stars):
            angle = (pygame.time.get_ticks() / 200.0) + (i * 2 * math.pi / num_stars)
            star_x = pos.x + math.cos(angle) * (radius + 5)
            star_y = pos.y - y_offset + math.sin(angle) * 3
            text = self.font_ui.render("zZ", True, self.COLOR_STUN_WAVE)
            self.screen.blit(text, (star_x - text.get_width()/2, star_y - text.get_height()/2))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Time Dilation
        time_text = self.font_ui.render(f"TIME: {self.time_dilation_factor:.1f}x", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 10))
        
        # Animal Selection
        animal_text_str = self.ANIMAL_NAMES[self.selected_animal]
        animal_text = self.font_animal.render(animal_text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(animal_text, (self.SCREEN_WIDTH/2 - animal_text.get_width()/2, self.SCREEN_HEIGHT - 35))

        # Cooldown indicator
        if self.ability_cooldown > 0:
            cooldown_text = self.font_ui.render(f"CD: {self.ability_cooldown:.1f}", True, (255, 100, 100))
            self.screen.blit(cooldown_text, (self.SCREEN_WIDTH/2 - cooldown_text.get_width()/2, self.SCREEN_HEIGHT - 55))

        # Game Over
        if self.game_over:
            if self.boss and self.boss['health'] <= 0:
                win_text = self.font_game_over.render("VICTORY!", True, self.COLOR_PLAYER)
                self.screen.blit(win_text, (self.SCREEN_WIDTH/2 - win_text.get_width()/2, self.SCREEN_HEIGHT/2 - win_text.get_height()/2))
            else:
                go_text = self.font_game_over.render("GAME OVER", True, self.COLOR_ENEMY)
                self.screen.blit(go_text, (self.SCREEN_WIDTH/2 - go_text.get_width()/2, self.SCREEN_HEIGHT/2 - go_text.get_height()/2))

    def close(self):
        pygame.quit()

# Example usage to run and visualize the environment
if __name__ == '__main__':
    # Un-dummy the video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Jungle Brawler")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    
    # --- Main Game Loop ---
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Keyboard Input for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset() # Automatically reset after game over
        
        # --- Rendering to Display ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        env.clock.tick(GameEnv.FPS)

    env.close()