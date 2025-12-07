import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:52:23.896061
# Source Brief: brief_02004.md
# Brief Index: 2004
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Match colored blocks to gather resources and deploy drones. "
        "Switch between demolition mode to destroy enemy skyscrapers and repair mode to defend your base."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the selector. Press space to match blocks. "
        "Press shift to switch between attack and repair modes."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_DEMOLITION = (255, 50, 50)
        self.COLOR_REPAIR = (50, 255, 100)
        self.COLOR_ENERGY = (80, 150, 255)
        self.COLOR_EXPLOSIVES = (255, 200, 50)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_GREY = (100, 100, 120)
        self.BLOCK_COLORS = {1: self.COLOR_ENERGY, 2: self.COLOR_EXPLOSIVES}

        # Game Grid
        self.GRID_ROWS, self.GRID_COLS = 8, 6
        self.GRID_X, self.GRID_Y = 220, 40
        self.BLOCK_SIZE = 36
        self.BLOCK_PADDING = 4
        
        # Gameplay
        self.BASE_MAX_HEALTH = 100
        self.SKYSCRAPER_MAX_HEALTH = 50
        self.NUM_SKYSCRAPERS = 2
        self.DRONE_COST_ENERGY = 25
        self.DRONE_COST_EXPLOSIVES = 25
        self.MAX_RESOURCES = 100
        self.DRONE_SPEED = 4
        self.DRONE_DAMAGE = 5
        self.DRONE_REPAIR_AMOUNT = 5
        
        # --- Gymnasium Interface ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        if self.render_mode == "human":
            pygame.display.set_caption("Drone Demolition")
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_base_health = None
        self.enemy_skyscrapers = None
        self.game_mode = None # 'demolition' or 'repair'
        self.energy_level = None
        self.explosives_level = None
        self.block_grid = None
        self.selector_pos = None
        self.drones = None
        self.particles = None
        self.match_animations = None
        self.enemy_attack_timer = None
        self.enemy_attack_frequency = None
        self.last_space_state = 0
        self.last_shift_state = 0
        
        # self.reset() # reset() is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_base_health = self.BASE_MAX_HEALTH
        self.enemy_skyscrapers = [{'health': self.SKYSCRAPER_MAX_HEALTH} for _ in range(self.NUM_SKYSCRAPERS)]
        
        self.game_mode = 'demolition'
        self.energy_level = 0
        self.explosives_level = 0
        
        self.block_grid = [[self.np_random.integers(1, len(self.BLOCK_COLORS) + 1) for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.selector_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        
        self.drones = []
        self.particles = []
        self.match_animations = []
        
        self.enemy_attack_timer = 0
        self.enemy_attack_frequency = 0.01

        self.last_space_state = 0
        self.last_shift_state = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0

        # 1. Handle Input
        step_reward += self._handle_input(action)

        # 2. Update Game State
        self._update_blocks()
        self._update_drones()
        self._update_enemy()
        self._update_particles()
        self._update_animations()

        # 3. Deploy Drones automatically
        step_reward += self._deploy_drones()

        # 4. Difficulty Scaling
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_attack_frequency = min(0.1, self.enemy_attack_frequency + 0.01)

        # 5. Check Termination & Calculate Final Reward
        terminated, terminal_reward = self._check_termination()
        self.game_over = terminated
        
        total_reward = step_reward + terminal_reward
        self.score += total_reward
        
        if self.render_mode == "human":
            self._render_human()
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            total_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action[0], action[1], action[2]
        reward = 0

        # Movement for selector
        if movement == 1: self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 2: self.selector_pos[0] = min(self.GRID_ROWS - 1, self.selector_pos[0] + 1)
        elif movement == 3: self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 4: self.selector_pos[1] = min(self.GRID_COLS - 1, self.selector_pos[1] + 1)

        # Space press (match blocks)
        if space_action == 1 and self.last_space_state == 0:
            # sfx: match_attempt.wav
            match_reward = self._find_and_process_matches(self.selector_pos[0], self.selector_pos[1])
            reward += match_reward
        self.last_space_state = space_action

        # Shift press (switch mode)
        if shift_action == 1 and self.last_shift_state == 0:
            # sfx: mode_switch.wav
            self.game_mode = 'repair' if self.game_mode == 'demolition' else 'demolition'
        self.last_shift_state = shift_action
        
        return reward

    def _find_and_process_matches(self, r, c):
        if r >= self.GRID_ROWS or c >= self.GRID_COLS or self.block_grid[r][c] is None:
            return 0

        color_to_match = self.block_grid[r][c]
        q = deque([(r, c)])
        matched_blocks = set([(r, c)])

        while q:
            curr_r, curr_c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                   self.block_grid[nr][nc] == color_to_match and (nr, nc) not in matched_blocks:
                    matched_blocks.add((nr, nc))
                    q.append((nr, nc))
        
        if len(matched_blocks) >= 3:
            # sfx: match_success.wav
            for r_m, c_m in matched_blocks:
                block_type = self.block_grid[r_m][c_m]
                if block_type == 1: # Energy
                    self.energy_level = min(self.MAX_RESOURCES, self.energy_level + 5)
                elif block_type == 2: # Explosives
                    self.explosives_level = min(self.MAX_RESOURCES, self.explosives_level + 5)
                
                # Add animation
                x = self.GRID_X + c_m * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
                y = self.GRID_Y + r_m * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
                self.match_animations.append({'pos': (x,y), 'radius': 0, 'max_radius': self.BLOCK_SIZE, 'timer': 10, 'color': self.BLOCK_COLORS[block_type]})
                
                self.block_grid[r_m][c_m] = None
            return 0.1 * len(matched_blocks)
        return 0

    def _update_blocks(self):
        # Gravity
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.block_grid[r][c] is not None:
                    if r != empty_row:
                        self.block_grid[empty_row][c] = self.block_grid[r][c]
                        self.block_grid[r][c] = None
                    empty_row -= 1
        
        # Refill
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.block_grid[r][c] is None:
                    self.block_grid[r][c] = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)

    def _deploy_drones(self):
        reward = 0
        if self.energy_level >= self.DRONE_COST_ENERGY and self.explosives_level >= self.DRONE_COST_EXPLOSIVES:
            self.energy_level -= self.DRONE_COST_ENERGY
            self.explosives_level -= self.DRONE_COST_EXPLOSIVES
            # sfx: drone_launch.wav
            
            start_pos = [100, self.HEIGHT / 2]
            target = None
            if self.game_mode == 'demolition':
                # Target the skyscraper with the most health
                alive_scrapers = [i for i, s in enumerate(self.enemy_skyscrapers) if s['health'] > 0]
                if alive_scrapers:
                    target_idx = max(alive_scrapers, key=lambda i: self.enemy_skyscrapers[i]['health'])
                    target_y = 60 + (self.HEIGHT - 120) * (target_idx / (self.NUM_SKYSCRAPERS -1 if self.NUM_SKYSCRAPERS > 1 else 1))
                    target = [self.WIDTH - 80, target_y]
            else: # repair
                target = [80, self.HEIGHT / 2]

            if target:
                self.drones.append({'pos': start_pos, 'target': target, 'mode': self.game_mode})
        return reward

    def _update_drones(self):
        reward = 0
        drones_to_remove = []
        for i, drone in enumerate(self.drones):
            pos = np.array(drone['pos'])
            target = np.array(drone['target'])
            direction = target - pos
            distance = np.linalg.norm(direction)

            if distance < self.DRONE_SPEED:
                drones_to_remove.append(i)
                if drone['mode'] == 'demolition':
                    # sfx: explosion.wav
                    self._create_explosion(drone['pos'], 30, 50)
                    # Find which skyscraper was the target
                    alive_scrapers = [s for s in self.enemy_skyscrapers if s['health'] > 0]
                    if alive_scrapers:
                        # Simple logic: damage the one with most health
                        target_scraper = max(alive_scrapers, key=lambda s: s['health'])
                        target_scraper['health'] = max(0, target_scraper['health'] - self.DRONE_DAMAGE)
                        reward += 0.5
                        if target_scraper['health'] == 0:
                            reward += 5 # Skyscraper destroyed bonus
                            # sfx: building_collapse.wav
                else: # repair
                    # sfx: repair.wav
                    self._create_repair_effect(drone['pos'], 20, 30)
                    self.player_base_health = min(self.BASE_MAX_HEALTH, self.player_base_health + self.DRONE_REPAIR_AMOUNT)
                    reward += 1.0
            else:
                drone['pos'] += (direction / distance) * self.DRONE_SPEED
        
        # Remove drones that reached their target
        for i in sorted(drones_to_remove, reverse=True):
            del self.drones[i]
        
        return reward

    def _update_enemy(self):
        self.enemy_attack_timer += self.enemy_attack_frequency
        if self.enemy_attack_timer >= 1.0:
            self.enemy_attack_timer -= 1.0
            self.player_base_health = max(0, self.player_base_health - 2)
            # sfx: incoming_fire.wav
            self._create_explosion([100, self.HEIGHT / 2], 20, 20, self.COLOR_DEMOLITION)

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _update_animations(self):
        anims_to_remove = []
        for i, anim in enumerate(self.match_animations):
            anim['timer'] -= 1
            anim['radius'] = self.BLOCK_SIZE/2 * (1 - (anim['timer']/10))
            if anim['timer'] <= 0:
                anims_to_remove.append(i)
        for i in sorted(anims_to_remove, reverse=True):
            del self.match_animations[i]

    def _check_termination(self):
        if self.player_base_health <= 0:
            return True, -100 # Loss
        if all(s['health'] <= 0 for s in self.enemy_skyscrapers):
            return True, 100 # Win
        return False, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_base_and_skyscrapers()
        self._render_block_grid()
        self._render_drones()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_base_health,
            "enemy_health": [s['health'] for s in self.enemy_skyscrapers],
            "mode": self.game_mode,
        }

    def _render_human(self):
        if self.human_screen is None: return
        obs = self._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        self.human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()

    # --- Rendering Helpers ---
    def _render_background(self):
        # Draw a subtle grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, (25, 30, 45), (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, (25, 30, 45), (0, y), (self.WIDTH, y))

    def _render_base_and_skyscrapers(self):
        # Player Base
        base_rect = pygame.Rect(30, 100, 50, self.HEIGHT - 200)
        pygame.draw.rect(self.screen, (60, 70, 90), base_rect, border_radius=5)
        pygame.gfxdraw.rectangle(self.screen, base_rect, (80, 90, 110))

        # Enemy Skyscrapers
        for i, s in enumerate(self.enemy_skyscrapers):
            y_pos = 60 + (self.HEIGHT - 120) * (i / (self.NUM_SKYSCRAPERS - 1 if self.NUM_SKYSCRAPERS > 1 else 1))
            height = 80
            width = 30
            rect = pygame.Rect(self.WIDTH - 60, y_pos - height / 2, width, height)
            
            health_ratio = s['health'] / self.SKYSCRAPER_MAX_HEALTH
            color = (int(80 - 40 * (1-health_ratio)), int(80 + 40 * health_ratio), 80)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.gfxdraw.rectangle(self.screen, rect, (120,120,120))
            
            # Damage overlay
            if health_ratio < 0.7:
                damage_surf = pygame.Surface((width, height), pygame.SRCALPHA)
                for _ in range(int(10 * (1 - health_ratio))):
                    start = (self.np_random.integers(0, width), self.np_random.integers(0, height))
                    end = (start[0] + self.np_random.integers(-5, 6), start[1] + self.np_random.integers(-5, 6))
                    pygame.draw.line(damage_surf, (0,0,0,150), start, end)
                self.screen.blit(damage_surf, rect.topleft)

            # Health text
            health_text = f"{int(s['health'])}%" if self.SKYSCRAPER_MAX_HEALTH == 100 else f"{int(s['health'])}"
            self._draw_text(health_text, (rect.centerx, rect.bottom + 10), self.font_s, self.COLOR_WHITE)

    def _render_block_grid(self):
        grid_w = self.GRID_COLS * self.BLOCK_SIZE
        grid_h = self.GRID_ROWS * self.BLOCK_SIZE
        grid_rect = pygame.Rect(self.GRID_X - self.BLOCK_PADDING, self.GRID_Y - self.BLOCK_PADDING, 
                                grid_w + self.BLOCK_PADDING*2, grid_h + self.BLOCK_PADDING*2)
        pygame.draw.rect(self.screen, (25, 30, 45), grid_rect, border_radius=5)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                block_type = self.block_grid[r][c]
                if block_type:
                    color = self.BLOCK_COLORS[block_type]
                    rect = pygame.Rect(self.GRID_X + c * self.BLOCK_SIZE, self.GRID_Y + r * self.BLOCK_SIZE,
                                       self.BLOCK_SIZE - self.BLOCK_PADDING, self.BLOCK_SIZE - self.BLOCK_PADDING)
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
                    
        # Render selector
        sel_r, sel_c = self.selector_pos
        sel_rect = pygame.Rect(self.GRID_X + sel_c * self.BLOCK_SIZE - 2, self.GRID_Y + sel_r * self.BLOCK_SIZE - 2,
                                self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, sel_rect, 2, border_radius=5)

        # Render match animations
        for anim in self.match_animations:
            pygame.gfxdraw.filled_circle(self.screen, int(anim['pos'][0]), int(anim['pos'][1]), int(anim['radius']), (*anim['color'], 150))

    def _render_drones(self):
        for drone in self.drones:
            color = self.COLOR_DEMOLITION if drone['mode'] == 'demolition' else self.COLOR_REPAIR
            pos = (int(drone['pos'][0]), int(drone['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_WHITE)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            # Create a temporary surface for the particle to handle alpha correctly
            particle_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, (*p['color'], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(particle_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

    def _render_ui(self):
        # Mode indicator border
        mode_color = self.COLOR_DEMOLITION if self.game_mode == 'demolition' else self.COLOR_REPAIR
        pygame.draw.rect(self.screen, mode_color, (0, 0, self.WIDTH, self.HEIGHT), 5)
        mode_text = f"MODE: {self.game_mode.upper()}"
        self._draw_text(mode_text, (self.WIDTH / 2, 20), self.font_m, self.COLOR_WHITE)

        # Player Base Health
        self._draw_text("BASE", (65, 85), self.font_m, self.COLOR_WHITE)
        self._draw_health_bar((30, self.HEIGHT - 95), (50, 15), self.player_base_health, self.BASE_MAX_HEALTH, self.COLOR_REPAIR)
        
        # Resources
        self._draw_text("ENERGY", (150, self.HEIGHT - 50), self.font_s, self.COLOR_ENERGY)
        self._draw_health_bar((150, self.HEIGHT - 35), (150, 15), self.energy_level, self.MAX_RESOURCES, self.COLOR_ENERGY)
        
        self._draw_text("EXPLOSIVES", (320, self.HEIGHT - 50), self.font_s, self.COLOR_EXPLOSIVES)
        self._draw_health_bar((320, self.HEIGHT - 35), (150, 15), self.explosives_level, self.MAX_RESOURCES, self.COLOR_EXPLOSIVES)

        # Score and Steps
        self._draw_text(f"SCORE: {int(self.score)}", (10, 10), self.font_s, self.COLOR_WHITE, align="topleft")
        self._draw_text(f"STEP: {self.steps}/{self.MAX_STEPS}", (self.WIDTH - 10, 10), self.font_s, self.COLOR_WHITE, align="topright")

    def _draw_text(self, text, position, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = position
        elif align == "topleft":
            text_rect.topleft = position
        elif align == "topright":
            text_rect.topright = position
        self.screen.blit(text_surface, text_rect)
    
    def _draw_health_bar(self, position, size, health, max_health, color):
        x, y = position
        w, h = size
        
        bg_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.COLOR_GREY, bg_rect, border_radius=3)
        
        fill_ratio = health / max(1, max_health)
        fill_w = int(w * fill_ratio)
        fill_rect = pygame.Rect(x, y, fill_w, h)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)
        
        pygame.draw.rect(self.screen, self.COLOR_WHITE, bg_rect, 1, border_radius=3)

    def _create_explosion(self, position, radius, count, color=None):
        if color is None:
            color = self.COLOR_EXPLOSIVES
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 31)
            self.particles.append({
                'pos': list(position),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(2, 5),
                'color': color
            })

    def _create_repair_effect(self, position, radius, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = self.np_random.uniform(0, radius)
            life = self.np_random.integers(20, 41)
            self.particles.append({
                'pos': [position[0] + math.cos(angle) * dist, position[1] + math.sin(angle) * dist],
                'vel': [0, -self.np_random.uniform(0.5, 1.5)],
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(3, 6),
                'color': self.COLOR_REPAIR
            })

if __name__ == '__main__':
    # To test with a random agent
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
    env.close()