import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:43:24.744704
# Source Brief: brief_03447.md
# Brief Index: 3447
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for a real-time strategy game.
    The player deploys a squad of units to defend a base against waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend your base by strategically deploying a squad of units against relentless waves of incoming enemies."
    user_guide = "Use arrow keys (↑↓←→) to move the cursor. Press space to deploy a unit. Press shift to command all units to attack."
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 19, 23)
    COLOR_GRID = (40, 45, 50)
    COLOR_BASE = (50, 80, 130)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_PLAYER = (60, 220, 120)
    COLOR_PLAYER_GLOW = (60, 220, 120, 50)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_GLOW = (255, 80, 80, 50)
    COLOR_HEALTH_BG = (70, 70, 70)
    COLOR_HEALTH_PLAYER = (60, 220, 120)
    COLOR_HEALTH_ENEMY = (255, 80, 80)
    COLOR_PROJECTILE_PLAYER = (180, 255, 200)
    COLOR_PROJECTILE_ENEMY = (255, 180, 180)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_TIMER = (255, 200, 0)

    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    GRID_AREA_WIDTH, GRID_AREA_HEIGHT = 512, 320
    CELL_WIDTH = GRID_AREA_WIDTH // GRID_WIDTH
    CELL_HEIGHT = GRID_AREA_HEIGHT // GRID_HEIGHT
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2 + 20

    # Game Parameters
    GAME_DURATION_SECONDS = 120.0
    TARGET_FPS = 60
    LOGIC_STEPS_PER_SECOND = 20
    MAX_EPISODE_STEPS = int(GAME_DURATION_SECONDS * LOGIC_STEPS_PER_SECOND)

    INITIAL_ENEMY_SPAWN_RATE = 0.2  # units per second
    SPAWN_RATE_INCREASE_INTERVAL = 30 # seconds
    SPAWN_RATE_INCREASE_AMOUNT = 0.1 # units per second

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

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
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_medium = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Game State Initialization ---
        self.cursor_pos = None
        self.player_units = None
        self.units_to_deploy = None
        self.enemy_units = None
        self.projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_outcome = None # "win", "loss", or ""
        self.timer = None
        self.enemy_spawn_timer = None
        self.current_spawn_rate = None
        self.last_spawn_increase_time = None
        self.last_space_held = None
        self.last_shift_held = None
        self.step_reward = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cursor_pos = np.array([self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2])
        self.player_units = []
        
        # Define the squad of 8 units
        squad_definitions = [
            # 4x Riflemen: standard range/damage
            {'type': 'rifleman', 'health': 100, 'range': 3.5, 'damage': 10, 'cooldown': 1.0, 'speed': 1.0},
            {'type': 'rifleman', 'health': 100, 'range': 3.5, 'damage': 10, 'cooldown': 1.0, 'speed': 1.0},
            {'type': 'rifleman', 'health': 100, 'range': 3.5, 'damage': 10, 'cooldown': 1.0, 'speed': 1.0},
            {'type': 'rifleman', 'health': 100, 'range': 3.5, 'damage': 10, 'cooldown': 1.0, 'speed': 1.0},
            # 2x Heavies: slow, tough, high damage
            {'type': 'heavy', 'health': 200, 'range': 2.5, 'damage': 25, 'cooldown': 2.0, 'speed': 0.5},
            {'type': 'heavy', 'health': 200, 'range': 2.5, 'damage': 25, 'cooldown': 2.0, 'speed': 0.5},
            # 2x Snipers: fragile, long range, slow fire rate
            {'type': 'sniper', 'health': 75, 'range': 6.0, 'damage': 40, 'cooldown': 3.0, 'speed': 0.8},
            {'type': 'sniper', 'health': 75, 'range': 6.0, 'damage': 40, 'cooldown': 3.0, 'speed': 0.8},
        ]
        self.units_to_deploy = deque(squad_definitions)

        self.enemy_units = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        self.timer = self.GAME_DURATION_SECONDS
        
        self.current_spawn_rate = self.INITIAL_ENEMY_SPAWN_RATE
        self.enemy_spawn_timer = 1.0 / self.current_spawn_rate if self.current_spawn_rate > 0 else float('inf')
        self.last_spawn_increase_time = self.GAME_DURATION_SECONDS

        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.step_reward = 0
        
        self._handle_input(action)
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        
        if terminated and not self.game_outcome: # Set outcome if not already set by win/loss
            if self.timer <= 0:
                self.game_outcome = "loss" # Time ran out
                self.step_reward -= 100

        self.steps += 1
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
            
        # --- Deploy Unit (on space press) ---
        if space_held and not self.last_space_held:
            self._deploy_unit()
            # sfx: UI_Confirm.wav

        # --- Trigger Attack (on shift press) ---
        if shift_held and not self.last_shift_held:
            self._trigger_global_attack()
            # sfx: UI_Click.wav
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
    def _update_game_state(self):
        delta_time = 1.0 / self.LOGIC_STEPS_PER_SECOND

        # Update timers
        self.timer = max(0, self.timer - delta_time)
        self.enemy_spawn_timer -= delta_time
        
        # Update spawn rate
        if self.last_spawn_increase_time - self.timer >= self.SPAWN_RATE_INCREASE_INTERVAL:
            self.current_spawn_rate += self.SPAWN_RATE_INCREASE_AMOUNT
            self.last_spawn_increase_time = self.timer

        # Update entities
        self._update_spawns()
        self._update_units(delta_time)
        self._update_projectiles(delta_time)
        self._update_particles(delta_time)

    def _deploy_unit(self):
        if not self.units_to_deploy: return
        
        # Can only deploy on the player's half of the grid (left side)
        if self.cursor_pos[0] >= self.GRID_WIDTH // 2: return
            
        # Check if a player unit already occupies the spot
        for unit in self.player_units:
            if np.array_equal(unit['grid_pos'], self.cursor_pos):
                return
        
        unit_def = self.units_to_deploy.popleft()
        new_unit = self._create_unit('player', self.cursor_pos, unit_def)
        self.player_units.append(new_unit)
        # sfx: Unit_Deploy.wav

    def _trigger_global_attack(self):
        for p_unit in self.player_units:
            if p_unit['attack_cooldown'] <= 0:
                target = self._find_closest_target(p_unit, self.enemy_units)
                if target:
                    self._create_projectile(p_unit, target)
                    p_unit['attack_cooldown'] = p_unit['cooldown']
                    # sfx: Player_Attack.wav

    def _update_spawns(self):
        if self.enemy_spawn_timer <= 0:
            spawn_y = self.np_random.integers(0, self.GRID_HEIGHT)
            spawn_pos = np.array([self.GRID_WIDTH - 1, spawn_y])
            
            # Ensure spawn location is not occupied
            if any(np.array_equal(e['grid_pos'], spawn_pos) for e in self.enemy_units):
                return

            enemy_def = {'type': 'grunt', 'health': 50, 'range': 1.5, 'damage': 10, 'cooldown': 1.5, 'speed': 1.0}
            new_enemy = self._create_unit('enemy', spawn_pos, enemy_def)
            self.enemy_units.append(new_enemy)
            
            self.enemy_spawn_timer += 1.0 / self.current_spawn_rate if self.current_spawn_rate > 0 else float('inf')
            # sfx: Enemy_Spawn.wav

    def _update_units(self, dt):
        # Update player units
        for unit in self.player_units:
            unit['attack_cooldown'] = max(0, unit['attack_cooldown'] - dt)
            unit['visual_pos'] += (unit['pixel_pos'] - unit['visual_pos']) * 0.15 # lerp

        # Update enemy units
        for unit in self.enemy_units:
            unit['attack_cooldown'] = max(0, unit['attack_cooldown'] - dt)
            
            # AI: Attack if possible, else move
            target = self._find_closest_target(unit, self.player_units)
            if target and unit['attack_cooldown'] <= 0:
                self._create_projectile(unit, target)
                unit['attack_cooldown'] = unit['cooldown']
                # sfx: Enemy_Attack.wav
            else:
                unit['move_cooldown'] -= dt
                if unit['move_cooldown'] <= 0:
                    old_pos = unit['grid_pos'].copy()
                    unit['grid_pos'][0] -= 1
                    unit['pixel_pos'] = self._grid_to_pixel(unit['grid_pos'])
                    unit['prev_pixel_pos'] = self._grid_to_pixel(old_pos)
                    unit['move_progress'] = 1.0
                    unit['move_cooldown'] += 1.0 / unit['speed']
            
            # Smooth movement interpolation
            unit['move_progress'] = max(0, unit['move_progress'] - dt * unit['speed'] * 2)
            unit['visual_pos'] = unit['prev_pixel_pos'] * unit['move_progress'] + unit['pixel_pos'] * (1 - unit['move_progress'])

        # Remove dead units
        self.player_units = [u for u in self.player_units if u['health'] > 0]
        self.enemy_units = [u for u in self.enemy_units if u['health'] > 0]

    def _update_projectiles(self, dt):
        for p in self.projectiles[:]:
            p['progress'] += dt * p['speed']
            if p['progress'] >= 1.0:
                if p['target']['health'] > 0: # Check if target is still alive
                    p['target']['health'] -= p['damage']
                    
                    # Reward shaping
                    if p['source_type'] == 'player':
                        self.step_reward += 0.1 # Damage dealt
                        if p['target']['health'] <= 0:
                            self.step_reward += 1.0 # Enemy killed
                            # sfx: Enemy_Explode.wav
                    else: # Enemy projectile
                        self.step_reward -= 0.2 # Damage taken
                        if p['target']['health'] <= 0:
                            # Player unit killed, no specific reward but it's a negative outcome
                            # sfx: Player_Explode.wav
                            pass
                    
                    self._create_particles(p['end_pos'], 15, p['color'])
                self.projectiles.remove(p)

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p['pos'] += p['vel'] * dt * 60 # Scale velocity to be independent of framerate
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= dt
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self):
        # Rewards are accumulated in self.step_reward during the step
        # Add terminal rewards here
        if self.game_over:
            if self.game_outcome == "win":
                self.step_reward += 100
            elif self.game_outcome == "loss":
                self.step_reward -= 100
        return self.step_reward

    def _check_termination(self):
        if self.game_over:
            return True

        # Loss Condition: Enemy reaches base (column 0)
        for enemy in self.enemy_units:
            if enemy['grid_pos'][0] < 1:
                self.game_over = True
                self.game_outcome = "loss"
                return True

        # Win Condition: Survived the full duration
        if self.timer <= 0:
            self.game_over = True
            self.game_outcome = "win"
            return True
            
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            # if base not breached, it's a win, otherwise loss is already set
            if not self.game_outcome:
                self.game_outcome = "win"
            return True

        return False

    def _get_observation(self):
        # Main render call
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "units_remaining": len(self.player_units),
            "enemies_on_screen": len(self.enemy_units),
            "deployable_units": len(self.units_to_deploy)
        }

    # --- Rendering Methods ---
    def _render_game(self):
        self._render_background()
        self._render_cursor()
        self._render_units()
        self._render_effects()

    def _render_background(self):
        # Draw base
        base_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.CELL_WIDTH, self.GRID_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, y))

    def _render_cursor(self):
        if self.game_over: return
        
        pos = self._grid_to_pixel(self.cursor_pos)
        rect = pygame.Rect(pos[0] - self.CELL_WIDTH // 2, pos[1] - self.CELL_HEIGHT // 2, self.CELL_WIDTH, self.CELL_HEIGHT)
        
        # Determine cursor color based on validity
        can_deploy = self.cursor_pos[0] < self.GRID_WIDTH // 2 and len(self.units_to_deploy) > 0
        color = self.COLOR_PLAYER if can_deploy else self.COLOR_ENEMY

        pygame.draw.rect(self.screen, color, rect, 2, border_radius=4)
        
        # Show attack range of all units when holding shift
        if self.last_shift_held:
            for unit in self.player_units:
                self._render_attack_range(unit)

    def _render_units(self):
        all_units = self.player_units + self.enemy_units
        for unit in sorted(all_units, key=lambda u: u['visual_pos'][1]):
            is_player = unit['owner'] == 'player'
            color = self.COLOR_PLAYER if is_player else self.COLOR_ENEMY
            glow_color = self.COLOR_PLAYER_GLOW if is_player else self.COLOR_ENEMY_GLOW
            
            pos = unit['visual_pos'].astype(int)
            radius = int(min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.35)
            
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 1.5), glow_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius * 1.5), glow_color)
            
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255,255,255))
            
            # Health bar
            health_pct = max(0, unit['health'] / unit['max_health'])
            bar_width = self.CELL_WIDTH * 0.8
            bar_height = 5
            bar_x = pos[0] - bar_width / 2
            bar_y = pos[1] - radius - 12
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_PLAYER if is_player else self.COLOR_HEALTH_ENEMY, (bar_x, bar_y, bar_width * health_pct, bar_height), border_radius=2)

    def _render_attack_range(self, unit):
        pos = unit['pixel_pos']
        radius = int(unit['range'] * self.CELL_WIDTH)
        
        # Create a transparent surface for the range circle
        range_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(range_surface, (50, 100, 200, 40), (radius, radius), radius)
        pygame.draw.circle(range_surface, (150, 200, 255, 60), (radius, radius), radius, 1)
        self.screen.blit(range_surface, (pos[0] - radius, pos[1] - radius))
        
    def _render_effects(self):
        # Projectiles
        for p in self.projectiles:
            start = p['start_pos']
            end = p['end_pos']
            pos = start + (end - start) * p['progress']
            pygame.draw.line(self.screen, p['color'], (int(pos[0]-5), int(pos[1])), (int(pos[0]+5), int(pos[1])), 3)

        # Particles
        for p in self.particles:
            size = max(1, int(p['size'] * (p['lifespan'] / p['max_lifespan'])))
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), size)

    def _render_ui(self):
        # Timer
        timer_text = f"{math.ceil(self.timer)}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_UI_TIMER)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH // 2 - timer_surf.get_width() // 2, 5))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))
        
        # Units to deploy
        deploy_text = f"DEPLOYABLE: {len(self.units_to_deploy)}"
        deploy_surf = self.font_medium.render(deploy_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(deploy_surf, (10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY" if self.game_outcome == "win" else "DEFEAT"
            color = self.COLOR_PLAYER if self.game_outcome == "win" else self.COLOR_ENEMY
            
            msg_surf = self.font_large.render(message, True, color)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_surf.get_height() // 2 - 20))

    # --- Helper Methods ---
    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + (grid_pos[0] + 0.5) * self.CELL_WIDTH
        y = self.GRID_OFFSET_Y + (grid_pos[1] + 0.5) * self.CELL_HEIGHT
        return np.array([x, y])
    
    def _create_unit(self, owner, grid_pos, definition):
        pixel_pos = self._grid_to_pixel(grid_pos)
        return {
            'id': self.np_random.random(),
            'owner': owner,
            'type': definition['type'],
            'grid_pos': grid_pos.copy(),
            'pixel_pos': pixel_pos,
            'visual_pos': pixel_pos.copy(),
            'prev_pixel_pos': pixel_pos.copy(),
            'move_progress': 0.0,
            'health': definition['health'],
            'max_health': definition['health'],
            'range': definition['range'],
            'damage': definition['damage'],
            'cooldown': definition['cooldown'],
            'speed': definition['speed'],
            'attack_cooldown': 0,
            'move_cooldown': 0
        }

    def _create_projectile(self, source, target):
        proj = {
            'start_pos': source['visual_pos'].copy(),
            'end_pos': target['visual_pos'].copy(),
            'progress': 0.0,
            'speed': 4.0, # in distance per second
            'damage': source['damage'],
            'source_type': source['owner'],
            'target': target,
            'color': self.COLOR_PROJECTILE_PLAYER if source['owner'] == 'player' else self.COLOR_PROJECTILE_ENEMY
        }
        self.projectiles.append(proj)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            lifespan = self.np_random.random() * 0.5 + 0.3
            self.particles.append({
                'pos': pos.copy().astype(float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': self.np_random.random() * 3 + 2
            })
            
    def _find_closest_target(self, source, target_list):
        closest_target = None
        min_dist_sq = (source['range'] * self.CELL_WIDTH) ** 2
        
        for target in target_list:
            dist_sq = np.sum((source['pixel_pos'] - target['pixel_pos']) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_target = target
        return closest_target

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("RTS Defense Environment")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        clock.tick(GameEnv.LOGIC_STEPS_PER_SECOND) # Control human play speed

    print(f"Game Over. Final Score: {info['score']}")
    env.close()