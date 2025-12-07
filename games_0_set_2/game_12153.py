import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:10:30.279554
# Source Brief: brief_02153.md
# Brief Index: 2153
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Explore a hazardous grid, collecting energy to terraform the landscape. "
        "Unlock teleport upgrades by reclaiming the world, but watch your energy levels."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport and press space to terraform the tile you are on."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (16, 16, 24)
    COLOR_GRID = (32, 32, 40)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_RESOURCE = (64, 192, 255)
    COLOR_HAZARD = (255, 64, 64)
    COLOR_HABITABLE = (64, 255, 128)
    COLOR_TEXT = (220, 220, 220)
    COLOR_ENERGY_BAR_FRAME = (80, 80, 90)
    COLOR_ENERGY_HIGH = (255, 255, 0)
    COLOR_ENERGY_LOW = (255, 0, 0)
    
    # Game Parameters
    INITIAL_ENERGY = 200
    MAX_ENERGY = 400
    ENERGY_PER_TELEPORT = 5
    ENERGY_PER_TERRAFORM = 20
    ENERGY_FROM_RESOURCE = 50
    HAZARD_ENERGY_DRAIN = 15
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # State variables are initialized in reset()
        self.grid = None
        self.resources = None
        self.player_pos = None
        self.player_pixel_pos = None
        self.player_target_pixel_pos = None
        self.player_angle = 0.0
        self.steps = 0
        self.score = 0
        self.energy = 0
        self.terraform_progress = 0.0
        self.total_terraformable_area = 0
        self.unlocked_abilities = None
        self.last_terraform_milestone = 0
        
        self.particles = []
        self.animations = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.energy = self.INITIAL_ENERGY
        self.last_terraform_milestone = 0
        self.unlocked_abilities = [False, False, False] # For 25%, 50%, 75%
        
        self._setup_level()
        
        # Place player on a safe, non-resource tile
        while True:
            start_pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if self.grid[start_pos[1], start_pos[0]] == 0 and start_pos not in self.resources:
                self.player_pos = start_pos
                break
        
        self.player_pixel_pos = np.array(self._grid_to_pixel(self.player_pos), dtype=float)
        self.player_target_pixel_pos = self.player_pixel_pos.copy()
        self.player_angle = 0.0
        
        self.particles.clear()
        self.animations.clear()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int) # 0: untamed, 1: habitable, 2: hazard
        self.total_terraformable_area = self.GRID_WIDTH * self.GRID_HEIGHT

        # Place hazards
        num_hazards = self.np_random.integers(15, 25)
        for _ in range(num_hazards):
            x, y = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            self.grid[y, x] = 2

        # Place resources
        self.resources = []
        num_resources = self.np_random.integers(20, 30)
        for _ in range(num_resources):
            while True:
                pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                if pos not in self.resources:
                    self.resources.append(pos)
                    break

    def step(self, action):
        movement, space_held, _ = action
        reward = 0
        terminated = False
        truncated = False

        # --- 1. Handle Actions and State Changes ---
        
        # A. Movement / Teleportation
        if movement != 0:
            teleport_distance = 1 + self.unlocked_abilities[0] + self.unlocked_abilities[1]
            
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1 # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1 # Right
            
            target_pos = (
                self.player_pos[0] + dx * teleport_distance,
                self.player_pos[1] + dy * teleport_distance
            )

            # Clamp to grid bounds
            target_pos = (
                max(0, min(self.GRID_WIDTH - 1, target_pos[0])),
                max(0, min(self.GRID_HEIGHT - 1, target_pos[1]))
            )

            if target_pos != self.player_pos:
                self.energy -= self.ENERGY_PER_TELEPORT
                reward -= 0.1 # Small cost for moving
                
                start_pixel = self._grid_to_pixel(self.player_pos)
                end_pixel = self._grid_to_pixel(target_pos)
                self._add_teleport_trail(start_pixel, end_pixel)
                
                self.player_pos = target_pos
                self.player_target_pixel_pos = np.array(end_pixel, dtype=float)

                # Check for hazards
                if self.grid[self.player_pos[1], self.player_pos[0]] == 2:
                    reward -= 1.0
                    self.energy -= self.HAZARD_ENERGY_DRAIN
                    self._add_particles(self.player_pixel_pos, 15, self.COLOR_HAZARD, 2.0, 4.0, 15)

        # B. Terraforming
        if space_held:
            px, py = self.player_pos
            if self.grid[py, px] != 1 and self.energy >= self.ENERGY_PER_TERRAFORM:
                self.energy -= self.ENERGY_PER_TERRAFORM
                self.grid[py, px] = 1
                self._add_terraform_animation((px, py))
        
        # C. Resource Collection
        if self.player_pos in self.resources:
            self.resources.remove(self.player_pos)
            self.energy = min(self.MAX_ENERGY, self.energy + self.ENERGY_FROM_RESOURCE)
            reward += 1.0
            self._add_particles(self.player_pixel_pos, 20, self.COLOR_RESOURCE, 1.0, 5.0, 20)

        # --- 2. Update Progression and Rewards ---
        
        # Terraform progress
        new_terraform_progress = np.count_nonzero(self.grid == 1) / self.total_terraformable_area
        self.terraform_progress = new_terraform_progress
        
        current_milestone = int(self.terraform_progress * 10) # 0-10
        if current_milestone > self.last_terraform_milestone:
            reward += 10.0 * (current_milestone - self.last_terraform_milestone)
            self.last_terraform_milestone = current_milestone

        # Ability unlocks
        if not self.unlocked_abilities[0] and self.terraform_progress >= 0.25:
            self.unlocked_abilities[0] = True
            reward += 5.0
        if not self.unlocked_abilities[1] and self.terraform_progress >= 0.50:
            self.unlocked_abilities[1] = True
            reward += 5.0
        if not self.unlocked_abilities[2] and self.terraform_progress >= 0.75:
            self.unlocked_abilities[2] = True
            reward += 5.0

        self.score += reward
        self.steps += 1
        
        # --- 3. Check Termination ---
        if self.energy <= 0:
            terminated = True
            reward = -10.0 # Override reward on termination
            self.score += reward
        elif self.steps >= self.MAX_STEPS:
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "energy": self.energy,
            "terraform_progress": self.terraform_progress,
        }

    # --- Helper & Rendering Methods ---

    def _render_game(self):
        self._update_and_render_animations()
        self._render_grid()
        self._render_resources()
        self._update_and_render_particles()
        self._render_player()

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                tile_type = self.grid[y, x]
                
                is_animated = any(anim['pos'] == (x, y) for anim in self.animations)

                if not is_animated:
                    if tile_type == 1: # Habitable
                        pygame.draw.rect(self.screen, self.COLOR_HABITABLE, rect, 1)
                    elif tile_type == 2: # Hazard
                        pygame.draw.rect(self.screen, self.COLOR_HAZARD, rect)
                
                pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_GRID)

    def _render_resources(self):
        for res_pos in self.resources:
            pixel_pos = self._grid_to_pixel(res_pos)
            
            # Pulsating size effect
            pulse = math.sin(self.steps * 0.1 + pixel_pos[0]) * 2 + 8
            
            points = []
            for i in range(8):
                angle = (i / 8.0) * 2 * math.pi + (self.steps * 0.05)
                px = pixel_pos[0] + math.cos(angle) * pulse
                py = pixel_pos[1] + math.sin(angle) * pulse
                points.append((int(px), int(py)))
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_RESOURCE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_RESOURCE)

    def _render_player(self):
        # Smooth interpolation
        self.player_pixel_pos += (self.player_target_pixel_pos - self.player_pixel_pos) * 0.4
        
        # Smooth rotation towards movement
        if np.linalg.norm(self.player_target_pixel_pos - self.player_pixel_pos) > 1:
            target_angle = math.atan2(
                self.player_target_pixel_pos[1] - self.player_pixel_pos[1],
                self.player_target_pixel_pos[0] - self.player_pixel_pos[0]
            )
            # Lerp angle
            angle_diff = (target_angle - self.player_angle + math.pi) % (2 * math.pi) - math.pi
            self.player_angle += angle_diff * 0.2
        
        pos = (int(self.player_pixel_pos[0]), int(self.player_pixel_pos[1]))
        size = 10
        
        # Glow effect
        glow_radius = int(size * (1.5 + 0.2 * math.sin(self.steps * 0.2)))
        glow_color = (*self.COLOR_PLAYER, 50)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player triangle
        points = [
            (pos[0] + math.cos(self.player_angle) * size, pos[1] + math.sin(self.player_angle) * size),
            (pos[0] + math.cos(self.player_angle + 2.2) * size * 0.7, pos[1] + math.sin(self.player_angle + 2.2) * size * 0.7),
            (pos[0] + math.cos(self.player_angle - 2.2) * size * 0.7, pos[1] + math.sin(self.player_angle - 2.2) * size * 0.7),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Energy Bar
        bar_width, bar_height = 200, 20
        bar_x, bar_y = self.SCREEN_WIDTH - bar_width - 10, 10
        
        energy_ratio = max(0, self.energy / self.MAX_ENERGY)
        fill_width = int(bar_width * energy_ratio)
        
        energy_color = tuple(int(c1 * energy_ratio + c2 * (1 - energy_ratio)) for c1, c2 in zip(self.COLOR_ENERGY_HIGH, self.COLOR_ENERGY_LOW))
        
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_FRAME, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, energy_color, (bar_x, bar_y, fill_width, bar_height))
        
        text_surf = self.font_main.render("ENERGY", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (bar_x - text_surf.get_width() - 10, bar_y))

        # Score and Terraforming Progress
        score_text = f"SCORE: {int(self.score)}"
        terraform_text = f"TERRAFORMED: {self.terraform_progress:.1%}"
        
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        terraform_surf = self.font_main.render(terraform_text, True, self.COLOR_TEXT)
        
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(terraform_surf, (10, 30))
        
        # Unlocked Abilities
        ability_text = "TELEPORT LVL: " + str(1 + sum(self.unlocked_abilities))
        ability_surf = self.font_small.render(ability_text, True, self.COLOR_TEXT)
        self.screen.blit(ability_surf, (10, self.SCREEN_HEIGHT - 25))

    def _grid_to_pixel(self, grid_pos):
        return (
            grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
            grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def _add_particles(self, pos, count, color, min_speed, max_speed, lifespan):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': velocity,
                'life': lifespan,
                'max_life': lifespan,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })
            
    def _add_teleport_trail(self, start_pos, end_pos):
        dist = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        if dist == 0: return
        num_particles = int(dist / 4)
        for i in range(num_particles):
            t = i / max(1, num_particles - 1)
            pos = [start_pos[j] * (1 - t) + end_pos[j] * t for j in range(2)]
            self.particles.append({
                'pos': pos,
                'vel': [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5)],
                'life': 10 + i,
                'max_life': 20,
                'color': self.COLOR_PLAYER,
                'size': 3
            })
            
    def _add_terraform_animation(self, grid_pos):
        self.animations.append({
            'type': 'terraform',
            'pos': grid_pos,
            'progress': 0,
            'duration': 20,
        })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, (*p['color'], alpha))

    def _update_and_render_animations(self):
        for anim in self.animations[:]:
            anim['progress'] += 1
            if anim['type'] == 'terraform':
                t = anim['progress'] / anim['duration']
                rect = pygame.Rect(anim['pos'][0] * self.CELL_SIZE, anim['pos'][1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                # Expanding circle effect
                radius = int(self.CELL_SIZE * 0.7 * t)
                center = rect.center
                alpha = int(255 * (1 - t))
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, (*self.COLOR_HABITABLE, alpha))
                
                # Fill the tile
                pygame.draw.rect(self.screen, self.COLOR_HABITABLE, rect)

            if anim['progress'] >= anim['duration']:
                self.animations.remove(anim)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and debugging, not part of the Gym environment
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'macOS', etc.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Arrow keys: Teleport
    # Space: Terraform
    # Q: Quit
    
    action = [0, 0, 0] # Start with no-op
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Terraform Grid")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True

        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render for human viewing
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for smooth viewing

    env.close()
    print(f"Game Over. Final Info: {info}")