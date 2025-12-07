import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:03:21.285259
# Source Brief: brief_03367.md
# Brief Index: 3367
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw

class GameEnv(gym.Env):
    """
    Boson Towers: A real-time strategy Gymnasium environment.

    The player controls a cursor to collect energy and build two types of towers:
    - Gun Turrets: Fire projectiles at enemy towers.
    - Chrono Towers: Do not attack, but create a field that slows enemy projectiles.

    The goal is to destroy all enemy towers before they destroy yours.
    Enemies spawn more frequently and with more health as the game progresses.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0:none, 1:up, 2:down, 3:left, 4:right)
    - actions[1]: Place Tower (1:held)
    - actions[2]: Cycle Tower Type (1:held)
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A real-time strategy game where you build towers to defend your base. "
        "Collect energy, place gun turrets and chrono towers, and destroy all enemy structures to win."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place the selected tower. "
        "Press Shift to cycle between tower types."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_ENERGY = (50, 255, 150)
    COLOR_ENERGY_GLOW = (50, 255, 150, 60)
    COLOR_CHRONO = (255, 255, 0)
    COLOR_CHRONO_GLOW = (255, 255, 0, 30)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR_BG = (40, 40, 40)
    COLOR_HEALTH_BAR_PLAYER = (0, 200, 100)
    COLOR_HEALTH_BAR_ENEMY = (200, 50, 50)

    # Game Parameters
    CURSOR_SPEED = 8
    INITIAL_ENERGY = 100
    ENERGY_PARTICLE_VALUE = 25
    ENERGY_PARTICLE_COUNT = 8
    ENERGY_PARTICLE_RADIUS = 5
    TOWER_PLACEMENT_COOLDOWN = 10 # frames
    MIN_TOWER_DISTANCE = 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.tower_types = [
            {'name': 'Gun Turret', 'cost': 50, 'hp': 100, 'damage': 5, 'cooldown': 45, 'range': 150, 'proj_speed': 4, 'type': 'gun'},
            {'name': 'Chrono Tower', 'cost': 75, 'hp': 50, 'range': 100, 'slow_factor': 0.4, 'type': 'chrono'}
        ]

        # These attributes are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_energy = 0
        self.player_towers = []
        self.enemy_towers = []
        self.projectiles = []
        self.energy_particles = []
        self.fx_particles = []
        self.cursor_pos = np.array([0.0, 0.0])
        self.selected_tower_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.enemy_spawn_timer = 0
        self.enemy_spawn_interval = 300 # 10 seconds at 30fps
        self.place_cooldown_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_energy = self.INITIAL_ENERGY
        self.selected_tower_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.place_cooldown_timer = 0

        self.player_towers = []
        self.enemy_towers = []
        self.projectiles = []
        self.energy_particles = []
        self.fx_particles = []

        self._spawn_tower('player', self.tower_types[0], np.array([100, self.HEIGHT / 2]))
        self._spawn_tower('enemy', self.tower_types[0], np.array([self.WIDTH - 100, self.HEIGHT / 2]))

        for _ in range(self.ENERGY_PARTICLE_COUNT):
            self._spawn_energy_particle()

        self.cursor_pos = np.array([self.WIDTH / 4, self.HEIGHT / 2], dtype=float)

        self.enemy_spawn_interval = 300
        self.enemy_spawn_timer = self.enemy_spawn_interval

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Input & Player Actions ---
        self._handle_input(action)

        # --- 2. Update Game State ---
        reward += self._update_energy_particles()
        self._update_enemies()
        reward += self._update_towers()
        self._update_projectiles()
        destruction_rewards = self._handle_collisions()
        reward += destruction_rewards
        self._update_fx_particles()

        if self.place_cooldown_timer > 0:
            self.place_cooldown_timer -= 1

        # --- 3. Check for Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if len(self.enemy_towers) == 0 and len(self.player_towers) > 0:
                reward += 100 # Win bonus
                self.score += 100
            elif len(self.player_towers) == 0:
                reward -= 100 # Loss penalty
                self.score -= 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        move_vec = np.array([0, 0], dtype=float)
        if movement == 1: move_vec[1] -= 1 # Up
        if movement == 2: move_vec[1] += 1 # Down
        if movement == 3: move_vec[0] -= 1 # Left
        if movement == 4: move_vec[0] += 1 # Right
        self.cursor_pos += move_vec * self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Action: Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.tower_types)
            # SFX: UI_Bleep

        # Action: Place tower (on press)
        if space_held and not self.last_space_held and self.place_cooldown_timer == 0:
            self._place_tower()
            self.place_cooldown_timer = self.TOWER_PLACEMENT_COOLDOWN

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _place_tower(self):
        selected_type = self.tower_types[self.selected_tower_idx]
        if self.player_energy >= selected_type['cost']:
            # Check for proximity to other towers
            can_place = True
            for t in self.player_towers + self.enemy_towers:
                if np.linalg.norm(self.cursor_pos - t['pos']) < self.MIN_TOWER_DISTANCE:
                    can_place = False
                    break
            
            if can_place:
                self.player_energy -= selected_type['cost']
                self._spawn_tower('player', selected_type, self.cursor_pos.copy())
                self._create_fx_explosion(self.cursor_pos, self.COLOR_PLAYER, 10, 15)
                # SFX: Tower_Place

    def _spawn_tower(self, team, tower_type_info, pos):
        tower = {
            'team': team,
            'pos': pos,
            'type': tower_type_info['type'],
            'max_hp': tower_type_info['hp'],
            'hp': tower_type_info['hp'],
            'range': tower_type_info['range'],
            'target': None,
            'anim_timer': random.randint(0, 30)
        }
        if tower['type'] == 'gun':
            tower['cooldown_max'] = tower_type_info['cooldown']
            tower['cooldown'] = 0
            tower['damage'] = tower_type_info['damage']
            tower['proj_speed'] = tower_type_info['proj_speed']
        elif tower['type'] == 'chrono':
            tower['slow_factor'] = tower_type_info['slow_factor']
        
        if team == 'player':
            self.player_towers.append(tower)
        else: # enemy
            # Scale enemy health
            scale_factor = 1.0 + (self.steps // 100) * 0.01
            tower['max_hp'] = int(tower['max_hp'] * scale_factor)
            tower['hp'] = tower['max_hp']
            self.enemy_towers.append(tower)

    def _update_energy_particles(self):
        reward = 0
        collected_particles = []
        for i, p in enumerate(self.energy_particles):
            if np.linalg.norm(self.cursor_pos - p['pos']) < 20:
                self.player_energy += self.ENERGY_PARTICLE_VALUE
                reward += 0.1
                self.score += 1
                collected_particles.append(i)
                self._create_fx_explosion(p['pos'], self.COLOR_ENERGY, 5, 10)
                # SFX: Collect_Energy
        
        for i in sorted(collected_particles, reverse=True):
            del self.energy_particles[i]
            self._spawn_energy_particle()
        return reward

    def _spawn_energy_particle(self):
        pos = np.array([random.uniform(50, self.WIDTH-50), random.uniform(50, self.HEIGHT-50)])
        self.energy_particles.append({'pos': pos, 'anim_timer': random.randint(0, 60)})

    def _update_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            pos = np.array([random.uniform(self.WIDTH/2 + 50, self.WIDTH - 50), random.uniform(50, self.HEIGHT - 50)])
            # Ensure not too close to other towers
            valid_pos = False
            attempts = 0
            while not valid_pos and attempts < 10:
                valid_pos = True
                for t in self.player_towers + self.enemy_towers:
                    if np.linalg.norm(pos - t['pos']) < self.MIN_TOWER_DISTANCE:
                        valid_pos = False
                        pos = np.array([random.uniform(self.WIDTH/2 + 50, self.WIDTH - 50), random.uniform(50, self.HEIGHT - 50)])
                        break
                attempts += 1
            if valid_pos:
                self._spawn_tower('enemy', self.tower_types[0], pos)

            # Decrease spawn interval over time
            self.enemy_spawn_interval = max(30, self.enemy_spawn_interval - (30 * 0.1)) # 0.1s reduction per 200 steps
            self.enemy_spawn_timer = int(self.enemy_spawn_interval)

    def _update_towers(self):
        enemy_projectiles_fired = 0
        all_towers = self.player_towers + self.enemy_towers
        for t in all_towers:
            t['anim_timer'] = (t['anim_timer'] + 1) % 60
            if t['type'] != 'gun':
                continue
            
            if t['cooldown'] > 0:
                t['cooldown'] -= 1
            
            # Find target
            targets = self.enemy_towers if t['team'] == 'player' else self.player_towers
            if not targets:
                t['target'] = None
                continue

            # Check if current target is still valid
            if t['target'] is not None:
                if t['target']['hp'] <= 0 or np.linalg.norm(t['pos'] - t['target']['pos']) > t['range']:
                    t['target'] = None
            
            # Find new target if needed
            if t['target'] is None:
                closest_target = None
                min_dist = float('inf')
                for target_candidate in targets:
                    dist = np.linalg.norm(t['pos'] - target_candidate['pos'])
                    if dist <= t['range'] and dist < min_dist:
                        min_dist = dist
                        closest_target = target_candidate
                t['target'] = closest_target
            
            # Fire projectile
            if t['target'] and t['cooldown'] == 0:
                t['cooldown'] = t['cooldown_max']
                direction = t['target']['pos'] - t['pos']
                norm = np.linalg.norm(direction)
                if norm > 0:
                    vel = (direction / norm) * t['proj_speed']
                    self.projectiles.append({
                        'pos': t['pos'].copy(), 'vel': vel, 'team': t['team'],
                        'damage': t['damage'], 'trail': []
                    })
                    # SFX: Laser_Shoot
                    if t['team'] == 'enemy':
                        enemy_projectiles_fired += 1
        
        return -0.01 * enemy_projectiles_fired

    def _update_projectiles(self):
        for p in self.projectiles:
            # Update trail
            p['trail'].append(p['pos'].copy())
            if len(p['trail']) > 5:
                p['trail'].pop(0)

            # Apply chrono tower slowdown
            slow_factor = 1.0
            if p['team'] == 'enemy':
                for t in self.player_towers:
                    if t['type'] == 'chrono' and np.linalg.norm(p['pos'] - t['pos']) < t['range']:
                        slow_factor = min(slow_factor, t['slow_factor'])
            
            p['pos'] += p['vel'] * slow_factor
    
    def _handle_collisions(self):
        reward = 0
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            targets = self.enemy_towers if p['team'] == 'player' else self.player_towers
            for t in targets:
                if np.linalg.norm(p['pos'] - t['pos']) < 15: # Collision radius
                    t['hp'] -= p['damage']
                    self._create_fx_explosion(p['pos'], self.COLOR_PLAYER if p['team'] == 'player' else self.COLOR_ENEMY, 3, 5)
                    projectiles_to_remove.append(i)
                    # SFX: Impact
                    break
            if not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                if i not in projectiles_to_remove:
                    projectiles_to_remove.append(i)
        
        # Remove projectiles that hit or went off-screen
        for i in sorted(list(set(projectiles_to_remove)), reverse=True):
            del self.projectiles[i]

        # Check for tower destruction
        towers_to_remove = {'player': [], 'enemy': []}
        for team_list, team_name in [(self.player_towers, 'player'), (self.enemy_towers, 'enemy')]:
            for i, t in enumerate(team_list):
                if t['hp'] <= 0:
                    towers_to_remove[team_name].append(i)
                    color = self.COLOR_PLAYER if team_name == 'player' else self.COLOR_ENEMY
                    self._create_fx_explosion(t['pos'], color, 20, 40)
                    # SFX: Explosion
                    if team_name == 'player':
                        reward -= 1.0
                        self.score -= 10
                    else: # enemy
                        reward += 1.0
                        self.score += 10
        
        for team_name, indices in towers_to_remove.items():
            team_list = self.player_towers if team_name == 'player' else self.enemy_towers
            for i in sorted(indices, reverse=True):
                del team_list[i]
        
        return reward

    def _create_fx_explosion(self, pos, color, num_particles, lifetime):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.fx_particles.append({
                'pos': pos.copy(), 'vel': vel, 'color': color,
                'lifetime': lifetime, 'max_lifetime': lifetime
            })

    def _update_fx_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.fx_particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.fx_particles[i]

    def _check_termination(self):
        win = len(self.enemy_towers) == 0 and self.steps > 1
        loss = len(self.player_towers) == 0
        self.game_over = win or loss
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "energy": self.player_energy}

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw energy particles
        for p in self.energy_particles:
            pulse = 1 + 0.2 * math.sin(p['anim_timer'] / 10.0)
            self._draw_glowing_circle(self.screen, p['pos'], self.ENERGY_PARTICLE_RADIUS * pulse, self.COLOR_ENERGY, self.COLOR_ENERGY_GLOW)

        # Draw chrono fields
        for t in self.player_towers:
            if t['type'] == 'chrono':
                pulse = 1 + 0.05 * math.sin(t['anim_timer'] / 5.0)
                radius = t['range'] * pulse
                alpha = 20 + 10 * math.sin(t['anim_timer'] / 5.0)
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, int(radius), int(radius), int(radius), (*self.COLOR_CHRONO, int(alpha)))
                self.screen.blit(s, (int(t['pos'][0] - radius), int(t['pos'][1] - radius)))

        # Draw towers
        for t in self.player_towers + self.enemy_towers:
            self._draw_tower(t)

        # Draw projectiles
        for p in self.projectiles:
            self._draw_projectile(p)
        
        # Draw FX particles
        for p in self.fx_particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color, p['pos'].astype(int), 2)

        # Draw cursor and placement preview
        self._draw_cursor()

    def _draw_tower(self, tower):
        pos = tower['pos'].astype(int)
        color = self.COLOR_PLAYER if tower['team'] == 'player' else self.COLOR_ENEMY
        glow_color = self.COLOR_PLAYER_GLOW if tower['team'] == 'player' else self.COLOR_ENEMY_GLOW
        
        self._draw_glowing_circle(self.screen, pos, 12, color, glow_color)

        if tower['type'] == 'gun':
            pygame.gfxdraw.box(self.screen, (pos[0]-8, pos[1]-8, 16, 16), color)
        elif tower['type'] == 'chrono':
            points = [
                (pos[0], pos[1] - 10),
                (pos[0] - 10, pos[1] + 7),
                (pos[0] + 10, pos[1] + 7)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Health bar
        bar_width = 30
        bar_height = 5
        bar_pos_x = pos[0] - bar_width // 2
        bar_pos_y = pos[1] - 25
        health_pct = max(0, tower['hp'] / tower['max_hp'])
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_pos_x, bar_pos_y, bar_width, bar_height))
        health_color = self.COLOR_HEALTH_BAR_PLAYER if tower['team'] == 'player' else self.COLOR_HEALTH_BAR_ENEMY
        pygame.draw.rect(self.screen, health_color, (bar_pos_x, bar_pos_y, bar_width * health_pct, bar_height))

    def _draw_projectile(self, p):
        pos = p['pos'].astype(int)
        color = self.COLOR_PLAYER if p['team'] == 'player' else self.COLOR_ENEMY
        # Draw trail
        for i, trail_pos in enumerate(reversed(p['trail'])):
            alpha = int(255 * (i / len(p['trail'])))
            trail_color = (*color, alpha)
            pygame.draw.circle(self.screen, trail_color, trail_pos.astype(int), 3)
        # Draw head
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, color)

    def _draw_cursor(self):
        pos = self.cursor_pos.astype(int)
        tower_type = self.tower_types[self.selected_tower_idx]
        
        # Determine placement validity for color
        can_place = self.player_energy >= tower_type['cost']
        if can_place:
            for t in self.player_towers + self.enemy_towers:
                if np.linalg.norm(self.cursor_pos - t['pos']) < self.MIN_TOWER_DISTANCE:
                    can_place = False
                    break
        
        preview_color = (*self.COLOR_PLAYER, 90) if can_place else (*self.COLOR_ENEMY, 90)

        # Draw placement preview
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        if tower_type['type'] == 'gun':
            pygame.gfxdraw.box(s, (pos[0]-8, pos[1]-8, 16, 16), preview_color)
            pygame.gfxdraw.aacircle(s, pos[0], pos[1], tower_type['range'], (*preview_color[:3], 30))
        elif tower_type['type'] == 'chrono':
            points = [(pos[0], pos[1]-10), (pos[0]-10, pos[1]+7), (pos[0]+10, pos[1]+7)]
            pygame.gfxdraw.filled_polygon(s, points, preview_color)
            pygame.gfxdraw.aacircle(s, pos[0], pos[1], tower_type['range'], (*self.COLOR_CHRONO, 30))
        self.screen.blit(s, (0,0))
        
        # Draw cursor crosshair
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (pos[0] - 10, pos[1]), (pos[0] + 10, pos[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (pos[0], pos[1] - 10), (pos[0], pos[1] + 10), 2)

    def _render_ui(self):
        # Energy
        energy_text = self.font_large.render(f"ENERGY: {self.player_energy}", True, self.COLOR_ENERGY)
        self.screen.blit(energy_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 40))

        # Selected Tower
        selected = self.tower_types[self.selected_tower_idx]
        tower_text = f"BUILD: {selected['name']} [COST: {selected['cost']}]"
        select_surf = self.font_large.render(tower_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(select_surf, (self.WIDTH - select_surf.get_width() - 10, 10))

        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 40))
        
        if self.game_over or self.steps >= self.MAX_STEPS:
            outcome_text = ""
            if len(self.enemy_towers) == 0 and len(self.player_towers) > 0:
                outcome_text = "VICTORY"
            elif len(self.player_towers) == 0:
                outcome_text = "DEFEAT"
            else:
                outcome_text = "TIME LIMIT REACHED"
            
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            font = pygame.font.SysFont("monospace", 72, bold=True)
            text_surf = font.render(outcome_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color):
        pos = pos.astype(int)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius*1.5), glow_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run when the environment is used by the test suite
    env = GameEnv()
    obs, info = env.reset()
    
    # We need to create a display for human interaction
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Boson Towers")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input to Action ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()