import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:47:36.901849
# Source Brief: brief_01199.md
# Brief Index: 1199
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for Crystal Cavern Defense.
    The player controls a cursor to mine crystals, build defenses, and survive waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Defend your cavern's base by mining crystals and building automated turrets to fend off waves of incoming enemies."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Hold space to mine crystals and hold shift to build defenses."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_WALL = (40, 30, 50)
    COLOR_BASE = (60, 50, 80)
    COLOR_BASE_BORDER = (120, 110, 150)
    COLOR_CURSOR = (255, 255, 255)
    
    COLOR_CRYSTAL_BLUE = (0, 150, 255)
    COLOR_CRYSTAL_RED = (255, 50, 50)
    
    COLOR_DEFENSE_BLUE = (50, 200, 255)
    COLOR_DEFENSE_RED = (255, 100, 100)
    
    COLOR_ENEMY = (200, 40, 150)
    COLOR_PROJECTILE_BLUE = (150, 220, 255)
    COLOR_PROJECTILE_RED = (255, 150, 150)
    
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEALTH = (40, 200, 40)
    COLOR_UI_HEALTH_BG = (200, 40, 40)

    # Game Parameters
    MAX_STEPS = 5000
    MAX_WAVES = 20
    PREPARATION_TIME = 300  # 10 seconds at 30 FPS
    WAVE_TIME = 900 # 30 seconds at 30 FPS
    
    PLAYER_MAX_HEALTH = 100
    CURSOR_SPEED = 8
    
    CRYSTAL_COUNT = 10
    CRYSTAL_MAX_HEALTH = 50
    CRYSTAL_RESPAWN_TIME = 600 # 20 seconds

    DEFENSE_BLUE_COST = {'blue': 10, 'red': 0}
    DEFENSE_BLUE_RANGE = 100
    DEFENSE_BLUE_DAMAGE = 2
    DEFENSE_BLUE_COOLDOWN = 30 # 1 second
    
    DEFENSE_RED_COST = {'blue': 15, 'red': 5}
    DEFENSE_RED_RANGE = 140
    DEFENSE_RED_DAMAGE = 5
    DEFENSE_RED_COOLDOWN = 45 # 1.5 seconds
    
    MINE_COOLDOWN_FRAMES = 5
    BUILD_COOLDOWN_FRAMES = 15

    RED_CRYSTAL_UNLOCK_WAVE = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.screen_width = self.WIDTH
        self.screen_height = self.HEIGHT

        # --- GYMNASIUM SPACES ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("consolas", 16)
        self.font_large = pygame.font.SysFont("consolas", 24)
        
        # --- STATE VARIABLES ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.PLAYER_MAX_HEALTH
        self.resources = {'blue': 0, 'red': 0}
        self.wave_number = 0
        self.game_phase = 'PREPARATION'
        self.phase_timer = 0
        self.cursor_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=float)
        self.selected_defense_type = 'blue'
        self.crystals = []
        self.defenses = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.cavern_walls = []
        self.mine_cooldown = 0
        self.build_cooldown = 0
        self.screen_flash_timer = 0
        
        self.total_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.total_reward = 0.0
        self.game_over = False
        
        self.player_health = self.PLAYER_MAX_HEALTH
        self.resources = {'blue': 20, 'red': 0} # Start with some blue
        self.wave_number = 0
        
        self.cursor_pos = np.array([self.screen_width / 2, self.HEIGHT - 50], dtype=float)
        self.selected_defense_type = 'blue'

        self.defenses.clear()
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self._generate_cavern()
        self._start_new_phase()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.steps += 1
        
        # Cooldowns
        if self.mine_cooldown > 0: self.mine_cooldown -= 1
        if self.build_cooldown > 0: self.build_cooldown -= 1
        if self.screen_flash_timer > 0: self.screen_flash_timer -= 1

        # Handle player input
        reward += self._handle_input(action)
        
        # Update game state
        reward += self._update_game_state()
        
        self.total_reward += reward
        
        # Check for termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if self.player_health <= 0:
                reward += -100.0  # Loss penalty
            elif self.wave_number > self.MAX_WAVES:
                reward += 100.0   # Victory bonus
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Core Logic Methods ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Cursor Movement ---
        move_vec = np.array([0, 0], dtype=float)
        if movement == 1: move_vec[1] = -1  # Up
        elif movement == 2: move_vec[1] = 1   # Down
        elif movement == 3: move_vec[0] = -1  # Left
        elif movement == 4: move_vec[0] = 1   # Right
        
        self.cursor_pos += move_vec * self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.screen_width - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.screen_height - 1)

        # --- Actions (only in PREPARATION phase) ---
        if self.game_phase == 'PREPARATION':
            # --- Mining ---
            if space_held and self.mine_cooldown == 0:
                mined_crystal = False
                for crystal in self.crystals:
                    if crystal['health'] > 0 and np.linalg.norm(self.cursor_pos - crystal['pos']) < 15:
                        crystal['health'] -= 5
                        resource_type = crystal['type']
                        self.resources[resource_type] += 1
                        reward += 0.1 # Reward for mining
                        self.mine_cooldown = self.MINE_COOLDOWN_FRAMES
                        # SFX: Mining hit
                        for _ in range(3):
                            self._create_particle(self.cursor_pos, crystal['color'], life=10, size=4)
                        mined_crystal = True
                        break
                if not mined_crystal:
                    pass # SFX: Mining miss/clank
            
            # --- Building ---
            if shift_held and self.build_cooldown == 0:
                # Determine which defense to build
                if self.wave_number >= self.RED_CRYSTAL_UNLOCK_WAVE:
                    self.selected_defense_type = 'red'
                else:
                    self.selected_defense_type = 'blue'

                cost = self.DEFENSE_BLUE_COST if self.selected_defense_type == 'blue' else self.DEFENSE_RED_COST
                
                can_afford = self.resources['blue'] >= cost['blue'] and self.resources['red'] >= cost['red']
                is_valid_location = self.cursor_pos[1] < self.HEIGHT - 80 # Cannot build in base area
                
                if can_afford and is_valid_location:
                    self.resources['blue'] -= cost['blue']
                    self.resources['red'] -= cost['red']
                    
                    new_defense = {
                        'pos': self.cursor_pos.copy(),
                        'type': self.selected_defense_type,
                        'cooldown': 0,
                        'target': None
                    }
                    self.defenses.append(new_defense)
                    reward += 0.5 # Reward for building
                    self.build_cooldown = self.BUILD_COOLDOWN_FRAMES
                    # SFX: Building placed
                    for _ in range(10):
                        color = self.COLOR_DEFENSE_BLUE if new_defense['type'] == 'blue' else self.COLOR_DEFENSE_RED
                        self._create_particle(self.cursor_pos, color, life=20, size=5, speed=3)

        return reward

    def _update_game_state(self):
        reward = 0.0
        
        # --- Phase Management ---
        self.phase_timer -= 1
        if self.phase_timer <= 0:
            self._start_new_phase()

        # --- Update Crystals (Respawn) ---
        for crystal in self.crystals:
            if crystal['health'] <= 0:
                crystal['respawn_timer'] -= 1
                if crystal['respawn_timer'] <= 0:
                    crystal['health'] = self.CRYSTAL_MAX_HEALTH
                    # SFX: Crystal respawn
                    for _ in range(15):
                        self._create_particle(crystal['pos'], crystal['color'], life=30, size=6, speed=2)

        # --- WAVE PHASE LOGIC ---
        if self.game_phase == 'WAVE':
            # --- Update Defenses (Targeting & Firing) ---
            for defense in self.defenses:
                if defense['cooldown'] > 0:
                    defense['cooldown'] -= 1
                else:
                    # Find closest enemy in range
                    target = None
                    min_dist = float('inf')
                    range_sq = (self.DEFENSE_BLUE_RANGE if defense['type'] == 'blue' else self.DEFENSE_RED_RANGE) ** 2
                    
                    for enemy in self.enemies:
                        dist_sq = np.sum((defense['pos'] - enemy['pos'])**2)
                        if dist_sq < range_sq and dist_sq < min_dist:
                            min_dist = dist_sq
                            target = enemy
                    
                    if target:
                        defense['target'] = target
                        self._fire_projectile(defense)
                        # SFX: Turret fire
                        defense['cooldown'] = self.DEFENSE_BLUE_COOLDOWN if defense['type'] == 'blue' else self.DEFENSE_RED_COOLDOWN

            # --- Update Projectiles ---
            for proj in self.projectiles[:]:
                if not proj['target'] or proj['target'] not in self.enemies:
                    self.projectiles.remove(proj)
                    continue
                
                direction = proj['target']['pos'] - proj['pos']
                dist = np.linalg.norm(direction)
                
                if dist < 10: # Collision
                    proj['target']['health'] -= proj['damage']
                    # SFX: Enemy hit
                    for _ in range(5):
                        self._create_particle(proj['pos'], (255, 255, 100), life=8, size=3)
                    self.projectiles.remove(proj)
                else:
                    proj['pos'] += (direction / dist) * 12 # Projectile speed

            # --- Update Enemies ---
            enemies_to_remove = []
            for enemy in self.enemies:
                if enemy['health'] <= 0:
                    enemies_to_remove.append(enemy)
                    reward += 1.0 # Reward for kill
                    # SFX: Enemy destroyed
                    for _ in range(20):
                        self._create_particle(enemy['pos'], self.COLOR_ENEMY, life=25, size=5, speed=4)
                    continue

                # Move towards player base
                target_pos = np.array([self.WIDTH / 2, self.HEIGHT - 40])
                direction = target_pos - enemy['pos']
                dist = np.linalg.norm(direction)

                if dist < 20: # Reached base
                    self.player_health -= 10
                    reward -= 5.0 # Penalty for taking damage
                    self.screen_flash_timer = 5
                    # SFX: Player base hit / alarm
                    enemies_to_remove.append(enemy)
                else:
                    enemy['pos'] += (direction / dist) * enemy['speed']
            
            self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
            
            # If all enemies are defeated, end wave early
            if not self.enemies:
                self.phase_timer = 0

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)
            if p['life'] <= 0:
                self.particles.remove(p)
                
        return reward
        
    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if self.wave_number > self.MAX_WAVES:
            self.game_over = True
            return True
        return False

    def _start_new_phase(self):
        if self.game_phase == 'WAVE':
            self.game_phase = 'PREPARATION'
            self.phase_timer = self.PREPARATION_TIME
        else: # Was PREPARATION
            self.game_phase = 'WAVE'
            self.wave_number += 1
            self.phase_timer = self.WAVE_TIME
            self._spawn_wave()
            
    def _spawn_wave(self):
        num_enemies = 2 + self.wave_number * 2
        enemy_speed = 1.0 + self.wave_number * 0.05
        enemy_health = 10 + self.wave_number * 2
        
        for _ in range(num_enemies):
            angle = random.uniform(0, math.pi) # Spawn in top half
            radius = self.WIDTH / 2 + random.uniform(20, 50)
            spawn_pos = np.array([
                self.WIDTH / 2 + math.cos(angle) * radius,
                self.HEIGHT / 2 - 50 + math.sin(angle) * radius
            ])
            
            # Ensure spawn is within bounds and not too close to center
            spawn_pos[0] = np.clip(spawn_pos[0], 20, self.WIDTH - 20)
            spawn_pos[1] = np.clip(spawn_pos[1], 20, self.HEIGHT / 2)

            self.enemies.append({
                'pos': spawn_pos,
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'anim_offset': random.uniform(0, 2 * math.pi)
            })

    # --- Helper Methods ---

    def _generate_cavern(self):
        self.cavern_walls.clear()
        for _ in range(5):
            points = []
            num_points = random.randint(5, 8)
            start_angle = random.uniform(0, 2 * math.pi)
            center_x = random.choice([self.WIDTH * 0.2, self.WIDTH * 0.8])
            center_y = random.choice([self.HEIGHT * 0.2, self.HEIGHT * 0.8])
            for i in range(num_points):
                angle = start_angle + (i / num_points) * 2 * math.pi
                dist = random.uniform(80, 150)
                points.append((center_x + math.cos(angle) * dist, center_y + math.sin(angle) * dist))
            self.cavern_walls.append(points)

        self.crystals.clear()
        for i in range(self.CRYSTAL_COUNT):
            is_red = self.wave_number >= self.RED_CRYSTAL_UNLOCK_WAVE and i < 3 # 3 red crystals
            pos = np.array([random.uniform(50, self.WIDTH - 50), random.uniform(50, self.HEIGHT - 100)])
            self.crystals.append({
                'pos': pos,
                'type': 'red' if is_red else 'blue',
                'color': self.COLOR_CRYSTAL_RED if is_red else self.COLOR_CRYSTAL_BLUE,
                'health': self.CRYSTAL_MAX_HEALTH,
                'max_health': self.CRYSTAL_MAX_HEALTH,
                'respawn_timer': 0
            })

    def _fire_projectile(self, defense):
        proj_type = defense['type']
        self.projectiles.append({
            'pos': defense['pos'].copy(),
            'type': proj_type,
            'target': defense['target'],
            'damage': self.DEFENSE_BLUE_DAMAGE if proj_type == 'blue' else self.DEFENSE_RED_DAMAGE
        })
        # Muzzle flash
        color = self.COLOR_PROJECTILE_BLUE if proj_type == 'blue' else self.COLOR_PROJECTILE_RED
        self._create_particle(defense['pos'], color, life=5, size=6)

    def _create_particle(self, pos, color, life, size, speed=2.0):
        angle = random.uniform(0, 2 * math.pi)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed * random.uniform(0.5, 1.5)
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel,
            'color': color,
            'life': life,
            'size': size * random.uniform(0.8, 1.2)
        })

    def _get_info(self):
        return {
            "score": self.total_reward,
            "steps": self.steps,
            "player_health": self.player_health,
            "wave": self.wave_number,
            "phase": self.game_phase,
            "resources_blue": self.resources['blue'],
            "resources_red": self.resources['red'],
            "enemies_left": len(self.enemies),
        }

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Screen flash on damage
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = 150 * (self.screen_flash_timer / 5)
            flash_surface.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surface, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Cavern Walls
        for wall in self.cavern_walls:
            pygame.gfxdraw.filled_polygon(self.screen, wall, self.COLOR_WALL)

        # Player Base
        base_rect = pygame.Rect(self.WIDTH/2 - 100, self.HEIGHT - 70, 200, 60)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE_BORDER, base_rect, width=2, border_radius=5)

        # Crystals
        for crystal in self.crystals:
            if crystal['health'] > 0:
                scale = crystal['health'] / crystal['max_health']
                size = 8 + 7 * scale
                points = []
                for i in range(6):
                    angle = (i / 6) * 2 * math.pi
                    offset = random.uniform(0.8, 1.2) if i % 2 == 0 else random.uniform(0.6, 1.0)
                    points.append((
                        crystal['pos'][0] + math.cos(angle) * size * offset,
                        crystal['pos'][1] + math.sin(angle) * size * offset
                    ))
                
                # Glow effect
                for i in range(3, 0, -1):
                    glow_color = (*crystal['color'], 20 * i)
                    pygame.gfxdraw.filled_polygon(self.screen, [(p[0], p[1]) for p in points], glow_color)
                
                pygame.gfxdraw.aapolygon(self.screen, points, crystal['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, crystal['color'])

        # Defenses
        for defense in self.defenses:
            pos_int = (int(defense['pos'][0]), int(defense['pos'][1]))
            color = self.COLOR_DEFENSE_BLUE if defense['type'] == 'blue' else self.COLOR_DEFENSE_RED
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 10, (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, (255,255,255))
            if defense['type'] == 'red':
                 pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 4, (255,255,255))


        # Projectiles
        for proj in self.projectiles:
            pos_int = (int(proj['pos'][0]), int(proj['pos'][1]))
            color = self.COLOR_PROJECTILE_BLUE if proj['type'] == 'blue' else self.COLOR_PROJECTILE_RED
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 4, (*color, 100))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 3, color)

        # Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            anim_sin = math.sin(self.steps * 0.2 + enemy['anim_offset'])
            size = 8 + anim_sin * 2
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(size+3), (*self.COLOR_ENEMY, 50))
            # Body
            pygame.gfxdraw.filled_trigon(self.screen, pos_int[0], pos_int[1]-int(size), pos_int[0]-int(size), pos_int[1]+int(size), pos_int[0]+int(size), pos_int[1]+int(size), self.COLOR_ENEMY)
            pygame.gfxdraw.aatrigon(self.screen, pos_int[0], pos_int[1]-int(size), pos_int[0]-int(size), pos_int[1]+int(size), pos_int[0]+int(size), pos_int[1]+int(size), self.COLOR_ENEMY)

        # Particles
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 20)) if p['life'] < 20 else 255
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['size']), color)

        # Cursor
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - 8, y), (x + 8, y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - 8), (x, y + 8), 1)

    def _render_ui(self):
        # Resources
        blue_text = self.font_small.render(f"Blue: {self.resources['blue']}", True, self.COLOR_CRYSTAL_BLUE)
        self.screen.blit(blue_text, (10, 10))
        if self.wave_number >= self.RED_CRYSTAL_UNLOCK_WAVE:
            red_text = self.font_small.render(f"Red: {self.resources['red']}", True, self.COLOR_CRYSTAL_RED)
            self.screen.blit(red_text, (10, 30))

        # Wave Info
        if self.game_phase == 'PREPARATION':
            timer_sec = self.phase_timer // self.metadata['render_fps']
            phase_text_str = f"PREPARE: {timer_sec}s"
        else: # WAVE
            phase_text_str = f"WAVE {self.wave_number} / {self.MAX_WAVES}"
        
        phase_text = self.font_large.render(phase_text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(phase_text, (self.WIDTH/2 - phase_text.get_width()/2, 10))

        # Player Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH if self.PLAYER_MAX_HEALTH > 0 else 0
        bar_width = 200
        bar_height = 15
        bar_x = self.WIDTH / 2 - bar_width / 2
        bar_y = self.HEIGHT - 90
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (bar_x, bar_y, bar_width * health_pct, bar_height), border_radius=3)
        health_text = self.font_small.render(f"{self.player_health}/{self.PLAYER_MAX_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (bar_x + bar_width/2 - health_text.get_width()/2, bar_y))
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Check if SDL_VIDEODRIVER is set to "dummy"
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"

    if not is_headless:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Crystal Cavern Defense")
    
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
        else: # In headless mode, take random actions
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if not is_headless:
            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            if not is_headless:
                pygame.time.wait(2000)
            running = False

        clock.tick(env.metadata["render_fps"])
        
    env.close()