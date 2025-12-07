import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:36:49.418123
# Source Brief: brief_02275.md
# Brief Index: 2275
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Signal Chain'.

    The player must draw interference patterns to disrupt an enemy communication
    network. The goal is to reduce all enemy signal strengths to zero before
    they collectively reach a critical threshold.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Disrupt an enemy communication network by drawing interference patterns. "
        "Weaken all enemy nodes to win before their signal strength overwhelms you."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to draw a jamming pattern. "
        "Press space to deploy the pattern and shift to switch between jammer types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (20, 40, 60)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_WEAK = (100, 25, 25)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_SUCCESS = (50, 255, 50)
    COLOR_UI_FAILURE = (255, 50, 50)
    COLOR_CURSOR = (255, 255, 255)

    # Game Parameters
    NUM_ENEMY_NODES = 4
    ENEMY_STRENGTH_GROWTH_INTERVAL = 200
    ENEMY_STRENGTH_GROWTH_AMOUNT = 0.05
    LOSS_THRESHOLD_PER_NODE = 90.0

    MAX_BANDWIDTH = 100.0
    BANDWIDTH_REGEN_RATE = 0.15
    DRAW_STEP_DISTANCE = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        self.jamming_devices = [
            {
                "name": "Line Jammer",
                "color": (0, 255, 255),
                "cost_per_segment": 2.0,
                "jamming_power": 0.25,
                "aoe_radius": 25,
                "lifetime": 150, # 5 seconds at 30fps
            },
            {
                "name": "Burst Jammer",
                "color": (255, 0, 255),
                "cost_per_segment": 4.0,
                "jamming_power": 0.6,
                "aoe_radius": 35,
                "lifetime": 90, # 3 seconds at 30fps
            }
        ]
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.enemy_nodes = []
        self.player_bandwidth = 0.0
        self.active_patterns = []
        self.particles = []
        self.drawing_cursor = (0, 0)
        self.current_path = []
        self.selected_device_idx = 0
        self.last_space_state = 0
        self.last_shift_state = 0
        self.difficulty_modifier = 1.0

        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.difficulty_modifier = 1.0

        self.enemy_nodes = []
        margin = 60
        for i in range(self.NUM_ENEMY_NODES):
            self.enemy_nodes.append({
                "pos": (
                    self.np_random.integers(margin, self.SCREEN_WIDTH - margin),
                    self.np_random.integers(margin, self.SCREEN_HEIGHT - margin)
                ),
                "strength": self.np_random.uniform(5.0, 15.0),
                "radius": 10
            })

        self.player_bandwidth = self.MAX_BANDWIDTH
        self.active_patterns = []
        self.particles = []
        self.drawing_cursor = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.current_path = [self.drawing_cursor]
        self.selected_device_idx = 0
        self.last_space_state = 0
        self.last_shift_state = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- 1. Process Player Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- 2. Update Game Logic ---
        self.player_bandwidth = min(self.MAX_BANDWIDTH, self.player_bandwidth + self.BANDWIDTH_REGEN_RATE)
        reward -= self.BANDWIDTH_REGEN_RATE * 0.01 # Small penalty for time passing / bandwidth gain

        self._update_enemy_signals()
        jamming_reward = self._update_jamming_patterns()
        reward += jamming_reward
        self._update_particles()

        # --- 3. Calculate Rewards & Termination ---
        total_enemy_strength = sum(node['strength'] for node in self.enemy_nodes)
        
        terminated = False
        truncated = False
        if total_enemy_strength >= self.LOSS_THRESHOLD_PER_NODE * self.NUM_ENEMY_NODES:
            reward -= 100
            terminated = True
            self.game_over = True
            self.win_state = False
        elif total_enemy_strength <= 0.01: # Use a small epsilon for float comparison
            reward += 100
            terminated = True
            self.game_over = True
            self.win_state = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Cycle jamming device on SHIFT press
        if shift_held and not self.last_shift_state:
            self.selected_device_idx = (self.selected_device_idx + 1) % len(self.jamming_devices)
            # SFX: UI_SWITCH

        # Commit pattern on SPACE press
        if space_held and not self.last_space_state and len(self.current_path) > 1:
            device = self.jamming_devices[self.selected_device_idx]
            self.active_patterns.append({
                "path": list(self.current_path),
                "device": device,
                "start_step": self.steps,
            })
            self.current_path = [(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)]
            self.drawing_cursor = self.current_path[0]
            # SFX: PATTERN_DEPLOY

        # Handle drawing movement
        if movement != 0:
            device = self.jamming_devices[self.selected_device_idx]
            cost = device["cost_per_segment"]
            if self.player_bandwidth >= cost:
                dx, dy = 0, 0
                if movement == 1: dy = -self.DRAW_STEP_DISTANCE  # Up
                elif movement == 2: dy = self.DRAW_STEP_DISTANCE   # Down
                elif movement == 3: dx = -self.DRAW_STEP_DISTANCE  # Left
                elif movement == 4: dx = self.DRAW_STEP_DISTANCE   # Right

                new_pos = (
                    np.clip(self.drawing_cursor[0] + dx, 0, self.SCREEN_WIDTH),
                    np.clip(self.drawing_cursor[1] + dy, 0, self.SCREEN_HEIGHT)
                )
                self.drawing_cursor = new_pos
                self.current_path.append(new_pos)
                self.player_bandwidth -= cost
        
        self.last_space_state = space_held
        self.last_shift_state = shift_held

    def _update_enemy_signals(self):
        if self.steps > 0 and self.steps % self.ENEMY_STRENGTH_GROWTH_INTERVAL == 0:
            self.difficulty_modifier += self.ENEMY_STRENGTH_GROWTH_AMOUNT

        for node in self.enemy_nodes:
            if node['strength'] > 0:
                node['strength'] = min(100.0, node['strength'] + 0.01 * self.difficulty_modifier)

    def _update_jamming_patterns(self):
        jamming_reward = 0
        patterns_to_remove = []
        for i, pattern in enumerate(self.active_patterns):
            device = pattern["device"]
            if self.steps > pattern["start_step"] + device["lifetime"]:
                patterns_to_remove.append(i)
                continue

            for node in self.enemy_nodes:
                if node['strength'] > 0:
                    for point in pattern["path"]:
                        dist = math.hypot(point[0] - node['pos'][0], point[1] - node['pos'][1])
                        if dist < device["aoe_radius"]:
                            prev_strength = node['strength']
                            node['strength'] = max(0.0, node['strength'] - device["jamming_power"])
                            reduction = prev_strength - node['strength']
                            if reduction > 0:
                                jamming_reward += reduction * 0.1 # Continuous reward for reduction
                                if node['strength'] == 0:
                                    jamming_reward += 1.0 # Bonus for silencing a node
                                    # SFX: NODE_SILENCED
                                    self._create_silence_particles(node['pos'])
                                break # Node is affected, move to next node
        
        # Remove expired patterns
        for i in sorted(patterns_to_remove, reverse=True):
            del self.active_patterns[i]

        return jamming_reward

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _create_silence_particles(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
                'lifetime': self.np_random.integers(20, 40),
                'color': self.COLOR_UI_SUCCESS,
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bandwidth": self.player_bandwidth,
            "total_enemy_strength": sum(n['strength'] for n in self.enemy_nodes),
            "win": self.win_state,
        }

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render enemy signal lines
        if len(self.enemy_nodes) > 1:
            for i in range(len(self.enemy_nodes)):
                for j in range(i + 1, len(self.enemy_nodes)):
                    node1 = self.enemy_nodes[i]
                    node2 = self.enemy_nodes[j]
                    avg_strength = (node1['strength'] + node2['strength']) / 2
                    if avg_strength > 0:
                        alpha = int(np.clip(avg_strength * 1.5, 0, 150))
                        color = (*self.COLOR_ENEMY, alpha)
                        pygame.draw.aaline(self.screen, color, node1['pos'], node2['pos'])

        # Render enemy nodes
        for node in self.enemy_nodes:
            if node['strength'] > 0:
                pulse = 1 + 0.1 * math.sin(self.steps * 0.1 + node['pos'][0])
                radius = int(node['radius'] * (node['strength'] / 100.0 + 0.5) * pulse)
                glow_radius = int(radius * 2.5)
                
                # Glow effect
                glow_alpha = int(np.clip(node['strength'] * 1.5, 20, 100))
                glow_color = (*self.COLOR_ENEMY, glow_alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(node['pos'][0]), int(node['pos'][1]), glow_radius, glow_color)
                
                # Core node
                main_color = pygame.Color.lerp(pygame.Color(self.COLOR_ENEMY_WEAK), pygame.Color(self.COLOR_ENEMY), node['strength']/100.0)
                pygame.gfxdraw.filled_circle(self.screen, int(node['pos'][0]), int(node['pos'][1]), radius, main_color)
                pygame.gfxdraw.aacircle(self.screen, int(node['pos'][0]), int(node['pos'][1]), radius, main_color)

        # Render active patterns
        for pattern in self.active_patterns:
            device = pattern['device']
            age = self.steps - pattern['start_step']
            alpha = int(255 * (1 - age / device['lifetime']))
            if alpha > 0 and len(pattern['path']) > 1:
                color = (*device['color'], alpha)
                pygame.draw.aalines(self.screen, color, False, pattern['path'])
                
                # AOE visualization for clarity
                aoe_alpha = int(alpha * 0.15)
                aoe_color = (*device['color'], aoe_alpha)
                for point in pattern['path']:
                    pygame.gfxdraw.filled_circle(self.screen, int(point[0]), int(point[1]), device['aoe_radius'], aoe_color)

        # Render current drawing path
        if len(self.current_path) > 1:
            device = self.jamming_devices[self.selected_device_idx]
            pygame.draw.aalines(self.screen, device['color'], False, self.current_path)
        
        # Render drawing cursor
        pulse = 2 + 2 * abs(math.sin(self.steps * 0.2))
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, self.drawing_cursor, 4)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, self.drawing_cursor, 4 + pulse, 1)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 40.0))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Bandwidth Bar
        bw_label_surf = self.font_small.render("BANDWIDTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(bw_label_surf, (self.SCREEN_WIDTH - 160, 10))
        bw_ratio = self.player_bandwidth / self.MAX_BANDWIDTH
        bar_color = self.jamming_devices[self.selected_device_idx]['color']
        self._draw_bar((self.SCREEN_WIDTH - 160, 28), (150, 15), bar_color, bw_ratio)

        # Enemy Strength Bar
        total_strength = sum(n['strength'] for n in self.enemy_nodes)
        max_strength = self.NUM_ENEMY_NODES * 100.0
        loss_strength = self.LOSS_THRESHOLD_PER_NODE * self.NUM_ENEMY_NODES
        
        str_label_surf = self.font_small.render("NETWORK STRESS", True, self.COLOR_UI_TEXT)
        self.screen.blit(str_label_surf, (self.SCREEN_WIDTH - 160, 55))
        str_ratio = total_strength / max_strength
        self._draw_bar((self.SCREEN_WIDTH - 160, 73), (150, 15), self.COLOR_ENEMY, str_ratio)
        # Draw loss threshold marker
        loss_marker_x = (self.SCREEN_WIDTH - 160) + int(150 * (loss_strength / max_strength))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (loss_marker_x, 71), (loss_marker_x, 73 + 17))


        # Current Device
        device = self.jamming_devices[self.selected_device_idx]
        device_surf = self.font_medium.render(f"DEVICE: {device['name']}", True, device['color'])
        self.screen.blit(device_surf, (10, self.SCREEN_HEIGHT - 35))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.win_state:
                msg = "NETWORK SILENCED"
                color = self.COLOR_UI_SUCCESS
            else:
                msg = "NETWORK OVERWHELMED"
                color = self.COLOR_UI_FAILURE
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_bar(self, pos, size, color, progress):
        x, y = pos
        w, h = size
        progress = np.clip(progress, 0, 1)
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (x, y, w, h))
        # Fill
        pygame.draw.rect(self.screen, color, (x, y, int(w * progress), h))
        # Border
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (x, y, w, h), 1)

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()


# Example of how to run the environment
if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the evaluation server.
    
    # Un-comment the line below to run with a display window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    
    # Pygame setup for display
    pygame.display.set_caption("Signal Chain")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")

    running = True
    while running:
        # Get keyboard input for manual control
        space_pressed = 0
        shift_pressed = 0
        movement = 0 # no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1

        action = [movement, space_pressed, shift_pressed]

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Win: {info['win']}. Press 'R' to reset.")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()
    pygame.quit()