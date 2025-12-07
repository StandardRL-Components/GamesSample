import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:33:51.249131
# Source Brief: brief_01074.md
# Brief Index: 1074
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import copy

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a side-scrolling shooter where the player
    deploys and controls evolving robots to defeat a boss mech. The game
    features a tile-based deployment system and a time-rewind mechanic.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Deploy and evolve robots on a grid to defeat a powerful boss mech. Use a time-rewind ability to gain a tactical advantage."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select a grid tile. Press space to deploy a robot. Press shift to rewind time."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    REWIND_STEPS = 10
    REWIND_HISTORY_LENGTH = 100

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_LOCKED = (40, 50, 70)
    COLOR_GRID_UNLOCKED = (60, 75, 105)
    COLOR_GRID_SELECT = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SHADOW = (10, 12, 20)

    COLOR_PLAYER = (0, 200, 100)
    COLOR_PLAYER_GLOW = (0, 200, 100, 50)
    COLOR_PLAYER_PROJ = (0, 255, 255)

    COLOR_BOSS = (200, 50, 50)
    COLOR_BOSS_GLOW = (200, 50, 50, 80)
    COLOR_BOSS_PROJ = (255, 100, 0)
    
    COLOR_HEALTH_BAR_BG = (70, 70, 90)
    COLOR_HEALTH_BAR = (220, 40, 40)
    
    COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (255, 50, 50)]

    # Game Parameters
    TILE_GRID_DIMS = (5, 3)  # 5 columns, 3 rows
    TILE_SIZE = 60
    TILE_PADDING = 10
    GRID_START_X = 50
    GRID_START_Y = 120

    BOSS_X = 550
    BOSS_Y = 200
    BOSS_MAX_HEALTH = 1000
    BOSS_ATTACK_COOLDOWN = 60 # Steps
    BOSS_ATTACK_DAMAGE_BASE = 1.0
    
    ROBOT_MAX = 5
    ROBOT_ATTACK_COOLDOWN = 45 # Steps
    ROBOT_ATTACK_POWER_BASE = 5.0
    ROBOT_ATTACK_POWER_GROWTH = 0.2
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_robots = []
        self.boss = {}
        self.projectiles = []
        self.boss_projectiles = []
        self.particles = []
        self.selected_tile = (0, 0)
        self.unlocked_tiles = set()
        self.rewind_charges = 0
        self.game_history = []
        self.space_was_held = False
        self.shift_was_held = False
        self.last_reward = 0.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_robots = []
        self.boss = {
            'pos': pygame.Vector2(self.BOSS_X, self.BOSS_Y),
            'health': self.BOSS_MAX_HEALTH,
            'max_health': self.BOSS_MAX_HEALTH,
            'attack_cooldown': self.BOSS_ATTACK_COOLDOWN,
            'attack_damage': self.BOSS_ATTACK_DAMAGE_BASE,
            'size': 50
        }
        self.projectiles = []
        self.boss_projectiles = []
        self.particles = []

        self.selected_tile = (0, 0) # col, row
        self.unlocked_tiles = {(c, r) for c in range(1) for r in range(self.TILE_GRID_DIMS[1])}
        
        self.rewind_charges = 3
        self.game_history = []
        self.space_was_held = False
        self.shift_was_held = False
        self.last_reward = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Store state for rewind ---
        self._store_history()

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held
        
        reward = 0

        # 1. Tile Selection
        if movement != 0:
            # Sfx: UI_Bleep
            col, row = self.selected_tile
            if movement == 1: row = (row - 1) % self.TILE_GRID_DIMS[1] # Up
            elif movement == 2: row = (row + 1) % self.TILE_GRID_DIMS[1] # Down
            elif movement == 3: col = (col - 1) % self.TILE_GRID_DIMS[0] # Left
            elif movement == 4: col = (col + 1) % self.TILE_GRID_DIMS[0] # Right
            self.selected_tile = (col, row)

        # 2. Deploy Robot
        if space_pressed:
            if self._can_deploy_robot():
                # Sfx: Deploy_Robot
                self._deploy_robot()

        # 3. Rewind Time
        if shift_pressed and self.rewind_charges > 0:
            # Sfx: Rewind_Sound
            self._rewind_time()
            # No other logic happens this step after a rewind
            self.last_reward = 0
        else:
            # --- Update Game Logic (if not rewound) ---
            self.steps += 1
            
            self._update_robots()
            self._update_boss()
            self._update_projectiles()
            reward += self._handle_collisions()
            self._update_particles()
            reward += self._check_progression()

            self.last_reward = reward

        self.space_was_held = space_held
        self.shift_was_held = shift_held
        
        # --- Termination and Final Rewards ---
        terminated = False
        if self.boss['health'] <= 0:
            # Sfx: Victory_Fanfare
            reward += 100
            terminated = True
            self._create_explosion(self.boss['pos'], 150, 100)
        elif len(self.player_robots) == 0 and len(self.unlocked_tiles) >= self.TILE_GRID_DIMS[0] * self.TILE_GRID_DIMS[1]:
            # Sfx: Defeat_Sound
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Game Logic Helpers ---

    def _update_robots(self):
        for robot in self.player_robots:
            robot['attack_cooldown'] -= 1
            if robot['attack_cooldown'] <= 0:
                # Sfx: Robot_Shoot
                robot['attack_cooldown'] = self.ROBOT_ATTACK_COOLDOWN
                robot['attack_power'] += self.ROBOT_ATTACK_POWER_GROWTH
                
                proj_start_pos = robot['pos'] + pygame.Vector2(robot['size'], 0)
                direction = (self.boss['pos'] - proj_start_pos).normalize()
                
                self.projectiles.append({
                    'pos': proj_start_pos,
                    'vel': direction * 8,
                    'power': robot['attack_power'],
                    'size': 4
                })
                self._create_muzzle_flash(proj_start_pos, self.COLOR_PLAYER_PROJ)

    def _update_boss(self):
        self.boss['attack_damage'] = self.BOSS_ATTACK_DAMAGE_BASE + 0.5 * (self.steps // 200)
        self.boss['attack_cooldown'] -= 1
        if self.boss['attack_cooldown'] <= 0 and self.player_robots:
            # Sfx: Boss_Shoot
            self.boss['attack_cooldown'] = self.BOSS_ATTACK_COOLDOWN
            target_robot = random.choice(self.player_robots)
            
            proj_start_pos = self.boss['pos'] - pygame.Vector2(self.boss['size'], 0)
            direction = (target_robot['pos'] - proj_start_pos).normalize()

            self.boss_projectiles.append({
                'pos': proj_start_pos,
                'vel': direction * 6,
                'damage': self.boss['attack_damage'],
                'size': 6
            })
            self._create_muzzle_flash(proj_start_pos, self.COLOR_BOSS_PROJ)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            if not self.screen.get_rect().collidepoint(proj['pos']):
                self.projectiles.remove(proj)
        
        for proj in self.boss_projectiles[:]:
            proj['pos'] += proj['vel']
            if not self.screen.get_rect().collidepoint(proj['pos']):
                self.boss_projectiles.remove(proj)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Boss
        for proj in self.projectiles[:]:
            if proj['pos'].distance_to(self.boss['pos']) < self.boss['size']:
                # Sfx: Hit_Impact_Heavy
                damage = proj['power']
                self.boss['health'] -= damage
                reward += damage * 0.1
                self._create_explosion(proj['pos'], 20, 15)
                self.projectiles.remove(proj)
        
        # Boss projectiles vs Robots
        for proj in self.boss_projectiles[:]:
            for robot in self.player_robots[:]:
                if proj['pos'].distance_to(robot['pos']) < robot['size']:
                    # Sfx: Hit_Impact_Light
                    self.player_robots.remove(robot)
                    self._create_explosion(robot['pos'], 30, 20)
                    if proj in self.boss_projectiles:
                        self.boss_projectiles.remove(proj)
                    break # Projectile hits one robot and is destroyed
        return reward

    def _check_progression(self):
        reward = 0
        health_percent = self.boss['health'] / self.boss['max_health']
        
        # Unlock 2nd column at 80% health
        if health_percent <= 0.8 and not (1,0) in self.unlocked_tiles:
            for r in range(self.TILE_GRID_DIMS[1]): self.unlocked_tiles.add((1, r))
            reward += 1
        # Unlock 3rd column at 60% health
        if health_percent <= 0.6 and not (2,0) in self.unlocked_tiles:
            for r in range(self.TILE_GRID_DIMS[1]): self.unlocked_tiles.add((2, r))
            reward += 1
        # Unlock 4th column at 40% health
        if health_percent <= 0.4 and not (3,0) in self.unlocked_tiles:
            for r in range(self.TILE_GRID_DIMS[1]): self.unlocked_tiles.add((3, r))
            reward += 1
        # Unlock 5th column at 20% health
        if health_percent <= 0.2 and not (4,0) in self.unlocked_tiles:
            for r in range(self.TILE_GRID_DIMS[1]): self.unlocked_tiles.add((4, r))
            reward += 1
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] *= 0.95
            if p['lifetime'] <= 0 or p['radius'] < 1:
                self.particles.remove(p)

    def _can_deploy_robot(self):
        col, row = self.selected_tile
        if (col, row) not in self.unlocked_tiles:
            return False
        if len(self.player_robots) >= self.ROBOT_MAX:
            return False
        
        tile_center = self._get_tile_center(col, row)
        for robot in self.player_robots:
            if robot['pos'] == tile_center:
                return False
        return True

    def _deploy_robot(self):
        col, row = self.selected_tile
        tile_center = self._get_tile_center(col, row)
        self.player_robots.append({
            'pos': tile_center,
            'attack_power': self.ROBOT_ATTACK_POWER_BASE,
            'attack_cooldown': self.ROBOT_ATTACK_COOLDOWN,
            'size': 15,
            'tile': (col, row)
        })

    def _store_history(self):
        state = {
            'steps': self.steps,
            'score': self.score,
            'player_robots': copy.deepcopy(self.player_robots),
            'boss': copy.deepcopy(self.boss),
            'projectiles': copy.deepcopy(self.projectiles),
            'boss_projectiles': copy.deepcopy(self.boss_projectiles),
            'unlocked_tiles': copy.deepcopy(self.unlocked_tiles),
            'rewind_charges': self.rewind_charges,
        }
        self.game_history.append(state)
        if len(self.game_history) > self.REWIND_HISTORY_LENGTH:
            self.game_history.pop(0)

    def _rewind_time(self):
        if len(self.game_history) > self.REWIND_STEPS:
            self.rewind_charges -= 1
            for _ in range(self.REWIND_STEPS):
                if len(self.game_history) > 1:
                    self.game_history.pop()
            
            last_state = self.game_history[-1]
            self.steps = last_state['steps']
            self.score = last_state['score']
            self.player_robots = last_state['player_robots']
            self.boss = last_state['boss']
            self.projectiles = last_state['projectiles']
            self.boss_projectiles = last_state['boss_projectiles']
            self.unlocked_tiles = last_state['unlocked_tiles']
            self.rewind_charges = last_state['rewind_charges']
            
            # Create rewind visual effect
            for _ in range(50):
                lifetime = 20
                self.particles.append({
                    'pos': pygame.Vector2(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                    'vel': pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                    'radius': random.uniform(2, 5),
                    'color': (200, 200, 255),
                    'lifetime': lifetime,
                    'initial_lifetime': lifetime
                })

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_tiles()
        self._render_particles()
        self._render_projectiles()
        self._render_robots()
        self._render_boss()

    def _render_tiles(self):
        for r in range(self.TILE_GRID_DIMS[1]):
            for c in range(self.TILE_GRID_DIMS[0]):
                rect = self._get_tile_rect(c, r)
                color = self.COLOR_GRID_UNLOCKED if (c,r) in self.unlocked_tiles else self.COLOR_GRID_LOCKED
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
        
        # Selected tile highlight
        sel_c, sel_r = self.selected_tile
        rect = self._get_tile_rect(sel_c, sel_r)
        pygame.draw.rect(self.screen, self.COLOR_GRID_SELECT, rect, 2, border_radius=5)

    def _render_robots(self):
        for robot in self.player_robots:
            pos = (int(robot['pos'].x), int(robot['pos'].y))
            size = int(robot['size'])
            power_glow = min(255, int((robot['attack_power'] - self.ROBOT_ATTACK_POWER_BASE) * 20))
            glow_color = (self.COLOR_PLAYER_GLOW[0], self.COLOR_PLAYER_GLOW[1], self.COLOR_PLAYER_GLOW[2], power_glow)
            
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 4, glow_color)
            
            # Main body
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)
            
            # "Eye"
            eye_pos = (pos[0] + size // 2, pos[1])
            pygame.gfxdraw.filled_circle(self.screen, eye_pos[0], eye_pos[1], size // 4, self.COLOR_BG)

    def _render_boss(self):
        pos = (int(self.boss['pos'].x), int(self.boss['pos'].y))
        size = int(self.boss['size'])
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 10, self.COLOR_BOSS_GLOW)
        
        # Body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_BOSS)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_BOSS)
        
        # Eye that "looks"
        angle = (self.steps * 0.05) % (2 * math.pi)
        eye_offset = pygame.Vector2(math.cos(angle), math.sin(angle)) * (size * 0.6)
        eye_pos = (int(pos[0] + eye_offset.x), int(pos[1] + eye_offset.y))
        pygame.gfxdraw.filled_circle(self.screen, eye_pos[0], eye_pos[1], size // 4, (255, 255, 100))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj['size'] + 2, (*self.COLOR_PLAYER_PROJ, 80))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj['size'], self.COLOR_PLAYER_PROJ)
        for proj in self.boss_projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj['size'] + 2, (*self.COLOR_BOSS_PROJ, 80))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj['size'], self.COLOR_BOSS_PROJ)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                alpha = int(150 * (p['lifetime'] / p['initial_lifetime']))
                color = (*p['color'], max(0, min(255, alpha)))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        # Boss Health Bar
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 20
        health_ratio = max(0, self.boss['health'] / self.boss['max_health'])
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, 20, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (20, 20, bar_width * health_ratio, bar_height), border_radius=5)
        
        # Score and Step Text
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self._draw_text_with_shadow(score_text, (20, 50))
        
        step_text = self.font_small.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self._draw_text_with_shadow(step_text, (self.SCREEN_WIDTH - step_text.get_width() - 20, 55))
        
        # Rewind charges
        rewind_text = self.font_large.render(f"REWIND: {self.rewind_charges}", True, self.COLOR_TEXT)
        self._draw_text_with_shadow(rewind_text, (20, 360))
        
        # Last reward indicator
        if self.last_reward != 0:
            color = (100, 255, 100) if self.last_reward > 0 else (255, 100, 100)
            reward_text = self.font_small.render(f"{self.last_reward:+.2f}", True, color)
            self._draw_text_with_shadow(reward_text, (160, 55), (0,0,0))
            
    def _draw_text_with_shadow(self, surface, pos, shadow_color=COLOR_SHADOW):
        self.screen.blit(surface, (pos[0] + 2, pos[1] + 2)) # Shadow
        self.screen.blit(surface, pos) # Text

    # --- Position and Effect Helpers ---
    
    def _get_tile_rect(self, col, row):
        x = self.GRID_START_X + col * (self.TILE_SIZE + self.TILE_PADDING)
        y = self.GRID_START_Y + row * (self.TILE_SIZE + self.TILE_PADDING)
        return pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)

    def _get_tile_center(self, col, row):
        rect = self._get_tile_rect(col, row)
        return pygame.Vector2(rect.centerx, rect.centery)

    def _create_explosion(self, pos, num_particles, max_speed):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed * 0.1
            lifetime = random.randint(20, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': random.uniform(2, 8),
                'color': random.choice(self.COLOR_EXPLOSION),
                'lifetime': lifetime,
                'initial_lifetime': lifetime
            })
            
    def _create_muzzle_flash(self, pos, color):
        for _ in range(5):
            angle = random.uniform(-0.5, 0.5)
            speed = random.uniform(2, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = random.randint(5, 10)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': random.uniform(2, 5),
                'color': color,
                'lifetime': lifetime,
                'initial_lifetime': lifetime
            })

    # --- Gymnasium Interface Helpers ---
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "boss_health": self.boss['health'],
            "robots": len(self.player_robots),
            "rewinds": self.rewind_charges
        }
    
    def render(self):
        # This method is not used in the standard gym loop but is useful for human play.
        if self.render_mode == "rgb_array":
            return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # We need a display to run the manual test
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Boss Takedown")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()