import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:37:06.368846
# Source Brief: brief_02445.md
# Brief Index: 2445
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a shifting quantum foam, strategically teleporting through
    wormholes to map its unpredictable topology and harvest entanglement resources.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting quantum foam by teleporting through wormholes to map its unpredictable topology. "
        "Manage your entanglement resources to reveal 90% of the area before they collapse."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a wormhole. Press space to teleport to the selected "
        "wormhole and reveal the surrounding area. Hold shift to perform a costly deep scan, temporarily "
        "increasing your view."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_START = (5, 0, 15)
    COLOR_BG_END = (20, 0, 40)
    COLOR_MAPPER = (255, 255, 255)
    COLOR_MAPPER_GLOW = (200, 200, 255)
    COLOR_FOAM_UNMAPPED = (0, 100, 120)
    COLOR_FOAM_MAPPED = (40, 60, 200)
    COLOR_WORMHOLE = (255, 255, 0)
    COLOR_WORMHOLE_SELECTED = (255, 100, 0)
    COLOR_ENTANGLEMENT_HIGH = (100, 255, 100)
    COLOR_ENTANGLEMENT_LOW = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)

    # Screen and World Dimensions
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 1280, 800
    CELL_SIZE = 16
    GRID_WIDTH = WORLD_WIDTH // CELL_SIZE
    GRID_HEIGHT = WORLD_HEIGHT // CELL_SIZE

    # Game Parameters
    MAX_STEPS = 1000
    VICTORY_PERCENTAGE = 0.90
    INITIAL_ENTANGLEMENT = 100.0
    TELEPORT_COST = 5.0
    DEEP_SCAN_COST = 8.0
    DEEP_SCAN_DURATION = 30  # steps
    FOAM_SHIFT_TURNS = 10
    NUM_WORMHOLES = 4
    MAP_RADIUS = 3 # cells
    DEEP_SCAN_RADIUS = 6 # cells

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        self.steps = 0
        self.turn_count = 0
        self.score = 0.0
        self.game_over = False
        self.entanglement = self.INITIAL_ENTANGLEMENT
        self.mapper_pos = pygame.Vector2(0, 0)
        self.map_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
        self.foam_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=float)
        self.wormholes = []
        self.selected_wormhole_index = 0
        self.deep_scan_timer = 0
        self.reward_milestones = set()
        self.victory = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.turn_count = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        self.entanglement = self.INITIAL_ENTANGLEMENT
        self.mapper_pos = pygame.Vector2(self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2)
        
        self.map_grid.fill(False)
        self._generate_foam()
        
        self._update_map()
        self._generate_wormholes()
        
        self.selected_wormhole_index = 0
        self.deep_scan_timer = 0
        self.reward_milestones = set()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        action_taken = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        old_mapped_count = np.sum(self.map_grid)

        # 1. Handle wormhole selection
        if movement in [1, 2, 3, 4]: # Up, Down, Left, Right
            # A simple way to cycle through selections
            if movement == 1 or movement == 4: # Up or Right
                self.selected_wormhole_index = (self.selected_wormhole_index + 1) % self.NUM_WORMHOLES
            else: # Down or Left
                self.selected_wormhole_index = (self.selected_wormhole_index - 1 + self.NUM_WORMHOLES) % self.NUM_WORMHOLES

        # 2. Handle deep scan
        if shift_held and self.entanglement > self.DEEP_SCAN_COST:
            # sfx: deep_scan_activate.wav
            self.entanglement -= self.DEEP_SCAN_COST
            self.deep_scan_timer = self.DEEP_SCAN_DURATION
            action_taken = True

        # 3. Handle teleport
        if space_held and self.entanglement > self.TELEPORT_COST:
            # sfx: teleport.wav
            self.entanglement -= self.TELEPORT_COST
            self.mapper_pos = self.wormholes[self.selected_wormhole_index].copy()
            self._update_map()
            self._generate_wormholes()
            self.selected_wormhole_index = 0
            action_taken = True
        
        if self.deep_scan_timer > 0:
            self.deep_scan_timer -= 1

        if action_taken:
            self.turn_count += 1
            if self.turn_count % self.FOAM_SHIFT_TURNS == 0:
                self._generate_foam() # sfx: foam_shift.wav

        # Ensure entanglement doesn't go below zero
        self.entanglement = max(0, self.entanglement)

        # Calculate reward
        new_mapped_count = np.sum(self.map_grid)
        mapped_percentage = new_mapped_count / self.map_grid.size
        
        # Reward for new territory
        reward += (new_mapped_count - old_mapped_count) * 0.1
        
        # Penalty for taking a turn
        if action_taken:
            reward -= 0.01

        # Milestone rewards
        current_milestone = int(mapped_percentage * 10) # 0-10
        if current_milestone > 0 and current_milestone not in self.reward_milestones:
            self.reward_milestones.add(current_milestone)
            reward += 1.0 # sfx: milestone_achieved.wav

        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            if self.victory:
                self.score += 100
                reward += 100
            else:
                self.score -= 100
                reward -= 100

        truncated = self.steps >= self.MAX_STEPS
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_termination(self):
        mapped_percentage = np.sum(self.map_grid) / self.map_grid.size
        if mapped_percentage >= self.VICTORY_PERCENTAGE:
            self.game_over = True
            self.victory = True
            # sfx: victory_fanfare.wav
        elif self.entanglement <= 0:
            self.game_over = True
            # sfx: failure_sound.wav
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        return self.game_over

    def _get_observation(self):
        self._render_background()
        self._render_foam()
        self._render_wormholes()
        self._render_player()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "entanglement": self.entanglement,
            "mapped_percentage": np.sum(self.map_grid) / self.map_grid.size,
        }

    # --- Helper Methods ---
    def _world_to_screen(self, pos, camera_offset):
        return pos - camera_offset

    def _generate_foam(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                self.foam_grid[x, y] = self.np_random.random()

    def _generate_wormholes(self):
        self.wormholes.clear()
        
        # Try to find unmapped locations first
        unmapped_candidates = []
        for i in range(100): # Try 100 times to find a good spot
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = self.np_random.uniform(100, 180)
            pos = self.mapper_pos + pygame.Vector2(dist * math.cos(angle), dist * math.sin(angle))
            pos.x = np.clip(pos.x, 0, self.WORLD_WIDTH - 1)
            pos.y = np.clip(pos.y, 0, self.WORLD_HEIGHT - 1)
            grid_x, grid_y = int(pos.x / self.CELL_SIZE), int(pos.y / self.CELL_SIZE)
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT and not self.map_grid[grid_x, grid_y]:
                unmapped_candidates.append(pos)
                if len(unmapped_candidates) >= self.NUM_WORMHOLES:
                    break
        
        if len(unmapped_candidates) >= self.NUM_WORMHOLES:
            self.wormholes = random.sample(unmapped_candidates, self.NUM_WORMHOLES)
        else:
            self.wormholes.extend(unmapped_candidates)

        # Fill remaining slots with random locations if needed
        while len(self.wormholes) < self.NUM_WORMHOLES:
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = self.np_random.uniform(80, 150)
            pos = self.mapper_pos + pygame.Vector2(dist * math.cos(angle), dist * math.sin(angle))
            pos.x = np.clip(pos.x, 0, self.WORLD_WIDTH - 1)
            pos.y = np.clip(pos.y, 0, self.WORLD_HEIGHT - 1)
            self.wormholes.append(pos)
            
    def _update_map(self):
        center_x, center_y = int(self.mapper_pos.x / self.CELL_SIZE), int(self.mapper_pos.y / self.CELL_SIZE)
        for dx in range(-self.MAP_RADIUS, self.MAP_RADIUS + 1):
            for dy in range(-self.MAP_RADIUS, self.MAP_RADIUS + 1):
                if dx*dx + dy*dy <= self.MAP_RADIUS*self.MAP_RADIUS:
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                        if not self.map_grid[x,y]:
                            self.map_grid[x, y] = True
                            # sfx: map_tile_reveal.wav

    # --- Rendering Methods ---
    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_foam(self):
        camera_offset = self.mapper_pos - pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        visibility_radius = self.DEEP_SCAN_RADIUS if self.deep_scan_timer > 0 else self.MAP_RADIUS
        
        center_gx, center_gy = int(self.mapper_pos.x / self.CELL_SIZE), int(self.mapper_pos.y / self.CELL_SIZE)
        render_radius = int(self.SCREEN_WIDTH / self.CELL_SIZE / 2) + 5

        for gx in range(max(0, center_gx - render_radius), min(self.GRID_WIDTH, center_gx + render_radius)):
            for gy in range(max(0, center_gy - render_radius), min(self.GRID_HEIGHT, center_gy + render_radius)):
                dist_sq = (gx - center_gx)**2 + (gy - center_gy)**2
                is_visible = dist_sq <= visibility_radius**2
                
                if self.map_grid[gx, gy] or is_visible:
                    world_pos = pygame.Vector2(gx * self.CELL_SIZE + self.CELL_SIZE/2, gy * self.CELL_SIZE + self.CELL_SIZE/2)
                    screen_pos = self._world_to_screen(world_pos, camera_offset)
                    
                    if -20 < screen_pos.x < self.SCREEN_WIDTH + 20 and -20 < screen_pos.y < self.SCREEN_HEIGHT + 20:
                        pulse = math.sin(self.steps * 0.05 + self.foam_grid[gx, gy] * 5)
                        
                        is_mapped = self.map_grid[gx, gy]
                        color = self.COLOR_FOAM_MAPPED if is_mapped else self.COLOR_FOAM_UNMAPPED
                        
                        alpha = 180 if is_mapped else (100 if is_visible else 40)
                        alpha += pulse * 20
                        alpha = np.clip(alpha, 0, 255)
                        
                        size = int(self.CELL_SIZE * 0.4 + pulse * 1.5)
                        
                        pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), size, (*color, alpha))

    def _render_wormholes(self):
        camera_offset = self.mapper_pos - pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        for i, pos in enumerate(self.wormholes):
            screen_pos = self._world_to_screen(pos, camera_offset)
            
            is_selected = (i == self.selected_wormhole_index)
            color = self.COLOR_WORMHOLE_SELECTED if is_selected else self.COLOR_WORMHOLE
            
            pulse = (math.sin(self.steps * 0.1 + i) + 1) / 2 # 0 to 1
            radius = int(12 + pulse * 4)
            
            for j in range(3):
                alpha = 100 + pulse * 100 if is_selected else 50 + pulse * 100
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), radius + j, (*color, alpha))
            
            if is_selected:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), 3, color)


    def _render_player(self):
        # Player is always at the center of the screen
        center_screen = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # Entanglement orb
        ent_ratio = self.entanglement / self.INITIAL_ENTANGLEMENT
        orb_color = [int(c1 * ent_ratio + c2 * (1 - ent_ratio)) for c1, c2 in zip(self.COLOR_ENTANGLEMENT_HIGH, self.COLOR_ENTANGLEMENT_LOW)]
        orb_radius = int(20 + 80 * ent_ratio)
        
        for i in range(orb_radius, 0, -4):
            alpha = 50 * (1 - (i / orb_radius))**2
            pygame.gfxdraw.filled_circle(self.screen, int(center_screen.x), int(center_screen.y), i, (*orb_color, alpha))
            
        # Mapper core
        glow_radius = int(8 + (math.sin(self.steps * 0.1) + 1) * 2)
        for i in range(glow_radius, 0, -1):
            alpha = 150 * (1 - (i / glow_radius))
            pygame.gfxdraw.filled_circle(self.screen, int(center_screen.x), int(center_screen.y), i, (*self.COLOR_MAPPER_GLOW, alpha))
        
        pygame.gfxdraw.filled_circle(self.screen, int(center_screen.x), int(center_screen.y), 4, self.COLOR_MAPPER)
        pygame.gfxdraw.aacircle(self.screen, int(center_screen.x), int(center_screen.y), 4, self.COLOR_MAPPER)

    def _render_text(self, text, pos, color, font, shadow_color=None, shadow_offset=(2,2)):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # Mapped Percentage
        mapped_percentage = np.sum(self.map_grid) / self.map_grid.size * 100
        map_text = f"MAPPED: {mapped_percentage:.1f}%"
        self._render_text(map_text, (self.SCREEN_WIDTH - 180, 10), self.COLOR_TEXT, self.font_ui, self.COLOR_TEXT_SHADOW)
        
        # Entanglement Level
        ent_text = f"ENTANGLEMENT: {self.entanglement:.1f}"
        text_width = self.font_ui.size(ent_text)[0]
        self._render_text(ent_text, (self.SCREEN_WIDTH/2 - text_width/2, self.SCREEN_HEIGHT - 30), self.COLOR_TEXT, self.font_ui, self.COLOR_TEXT_SHADOW)

        # Deep Scan Indicator
        if self.deep_scan_timer > 0:
            scan_text = "DEEP SCAN ACTIVE"
            scan_width = self.font_ui.size(scan_text)[0]
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            color = (200 + 55 * pulse, 200 + 55 * pulse, 255)
            self._render_text(scan_text, (10, 10), color, self.font_ui, self.COLOR_TEXT_SHADOW)

        # Game Over / Victory Message
        if self.game_over:
            if self.victory:
                msg = "REGION MAPPED"
                color = self.COLOR_ENTANGLEMENT_HIGH
            else:
                msg = "ENTANGLEMENT COLLAPSE"
                color = self.COLOR_ENTANGLEMENT_LOW
            
            msg_width, msg_height = self.font_msg.size(msg)
            self._render_text(msg, (self.SCREEN_WIDTH/2 - msg_width/2, self.SCREEN_HEIGHT/2 - msg_height/2), color, self.font_msg, self.COLOR_TEXT_SHADOW)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This section is for human play and debugging, and will not be used by the evaluator.
    # It requires a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Quantum Foam Mapper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0.0
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    print("----------------\n")

    while not done:
        # --- Human Input ---
        movement_action = 0 # none
        space_action = 0 # released
        shift_action = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Mapped: {info['mapped_percentage']*100:.1f}%")

        # --- Render to screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth human play

    env.close()