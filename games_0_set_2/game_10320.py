import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:11:20.629401
# Source Brief: brief_00320.md
# Brief Index: 320
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Command a fleet of ships to harvest fuel from and capture nebulae in a strategic, turn-based space conquest."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a target nebula. Press 'space' to move or harvest. "
        "Hold 'shift' and press 'space' to upgrade. Press 'shift' alone to cycle ships."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_STAR = (200, 200, 220)
    COLOR_SHIP = (60, 180, 255)
    COLOR_SHIP_GLOW = (120, 220, 255)
    COLOR_NEUTRAL_NEBULA = (255, 80, 80)
    COLOR_PLAYER_NEBULA = (80, 255, 80)
    COLOR_FUEL = (255, 220, 50)
    COLOR_UI_BG = (20, 15, 40, 200)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_TARGET_LINE = (255, 255, 255, 100)
    
    # Game Parameters
    MAX_STEPS = 500
    NUM_NEBULAE = 12
    NUM_STARS = 150
    INITIAL_FUEL = 2000
    MAX_FUEL = 5000
    INITIAL_SHIPS = 2
    
    # Ship Parameters
    SHIP_UPGRADE_COST = 500
    SHIP_MAX_LEVEL = 5
    
    # Nebula Parameters
    NEBULA_MIN_FUEL = 50
    NEBULA_MAX_FUEL = 500
    NEBULA_REGEN_RATE = 0.2
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.font_medium = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.player_fuel = 0
        self.game_over = False
        
        self.nebulae = []
        self.ships = []
        self.stars = []
        self.particles = []
        
        self.selected_ship_idx = 0
        
        # UI Feedback
        self.action_feedback_text = ""
        self.action_feedback_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.player_fuel = self.INITIAL_FUEL
        self.game_over = False
        
        self._generate_stars()
        self._generate_nebulae()
        self._generate_ships()
        
        self.particles = []
        self.selected_ship_idx = 0
        self._update_ship_target(0) # Set initial target for first ship
        
        self.action_feedback_text = ""
        self.action_feedback_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input and Actions ---
        self.action_feedback_text = "" # Clear previous feedback
        
        selected_ship = self.ships[self.selected_ship_idx]
        
        # Action: Cycle selected ship (Shift without Space)
        if shift_pressed and not space_pressed:
            self.selected_ship_idx = (self.selected_ship_idx + 1) % len(self.ships)
            self._update_ship_target(self.selected_ship_idx)
            self.action_feedback_text = f"SELECTED SHIP {self.selected_ship_idx + 1}"
            self.action_feedback_timer = 30
        
        # Action: Change target nebula (Movement arrows)
        if movement > 0:
            self._find_next_target(movement)
        
        # Action: Execute (Space)
        if space_pressed:
            target_nebula = self.nebulae[selected_ship['target_id']]
            
            # Sub-Action: Upgrade (Shift + Space)
            if shift_pressed:
                if self.player_fuel >= self.SHIP_UPGRADE_COST and selected_ship['level'] < self.SHIP_MAX_LEVEL:
                    self.player_fuel -= self.SHIP_UPGRADE_COST
                    selected_ship['level'] += 1
                    # No direct reward for upgrading, it's a strategic cost
                    self.action_feedback_text = f"SHIP {self.selected_ship_idx + 1} UPGRADED!"
                else:
                    self.action_feedback_text = "UPGRADE FAILED"
                self.action_feedback_timer = 30
            
            # Sub-Action: Move or Harvest (Space only)
            else:
                ship_pos = np.array(selected_ship['pos'])
                nebula_pos = np.array(target_nebula['pos'])
                distance = np.linalg.norm(ship_pos - nebula_pos)
                
                if distance > 5: # Move
                    cost = int(distance * 0.5)
                    if self.player_fuel >= cost:
                        self.player_fuel -= cost
                        reward -= cost * 0.1
                        selected_ship['pos'] = target_nebula['pos']
                        self.action_feedback_text = f"MOVING... (-{cost} FUEL)"
                    else:
                        self.action_feedback_text = "INSUFFICIENT FUEL TO MOVE"
                    self.action_feedback_timer = 30
                else: # Harvest
                    was_neutral = target_nebula['owner'] == 'neutral'
                    harvest_amount = min(target_nebula['fuel'], 20 * selected_ship['level'])
                    
                    if harvest_amount > 0:
                        target_nebula['fuel'] -= harvest_amount
                        self.player_fuel = min(self.MAX_FUEL, self.player_fuel + harvest_amount)
                        reward += harvest_amount * 1.0 # +1 per fuel
                        
                        if was_neutral:
                            target_nebula['owner'] = 'player'
                            reward += 5 # +5 for capture
                            self.score += 1
                            self.action_feedback_text = f"CAPTURED & HARVESTED (+{int(harvest_amount)} FUEL)"
                        else:
                            self.action_feedback_text = f"HARVESTING (+{int(harvest_amount)} FUEL)"
                        
                        # Create harvest particles
                        for _ in range(int(harvest_amount / 2)):
                            self._create_particle(target_nebula['pos'], self.COLOR_FUEL, selected_ship['pos'])
                    else:
                        self.action_feedback_text = "NEBULA DEPLETED"
                    self.action_feedback_timer = 30

        # --- Update Game State ---
        self._update_nebulae()
        self._update_particles()
        if self.action_feedback_timer > 0:
            self.action_feedback_timer -= 1
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True # Truncated because time limit reached, not a failure state
            self.game_over = True
            if self.player_fuel > 0: # Bonus reward only if not lost by fuel
                final_bonus = sum(50 for n in self.nebulae if n['owner'] == 'player')
                reward += final_bonus
        elif self.player_fuel <= 0:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.player_fuel,
            "controlled_nebulae": sum(1 for n in self.nebulae if n['owner'] == 'player'),
        }

    # --- Rendering Methods ---
    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, star['pos'], star['size'])
            
    def _render_game(self):
        # Draw target line
        selected_ship = self.ships[self.selected_ship_idx]
        target_nebula = self.nebulae[selected_ship['target_id']]
        pygame.draw.aaline(self.screen, self.COLOR_TARGET_LINE, selected_ship['pos'], target_nebula['pos'], 1)

        # Draw Nebulae
        for nebula in self.nebulae:
            color = self.COLOR_PLAYER_NEBULA if nebula['owner'] == 'player' else self.COLOR_NEUTRAL_NEBULA
            pulse = (math.sin(self.steps * 0.05 + nebula['phase']) + 1) / 2
            
            # Glow
            glow_radius = int(nebula['size'] * (1.2 + pulse * 0.4))
            glow_alpha = int(50 + pulse * 40)
            self._draw_glowing_circle(nebula['pos'], glow_radius, color, glow_alpha)
            
            # Core
            pygame.gfxdraw.filled_circle(self.screen, int(nebula['pos'][0]), int(nebula['pos'][1]), int(nebula['size']), color)
            pygame.gfxdraw.aacircle(self.screen, int(nebula['pos'][0]), int(nebula['pos'][1]), int(nebula['size']), color)

        # Draw Ships
        for i, ship in enumerate(self.ships):
            # Ship Body
            p1 = (ship['pos'][0], ship['pos'][1] - 8)
            p2 = (ship['pos'][0] - 6, ship['pos'][1] + 6)
            p3 = (ship['pos'][0] + 6, ship['pos'][1] + 6)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_SHIP)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_SHIP)

            # Selection Highlight
            if i == self.selected_ship_idx:
                self._draw_glowing_circle(ship['pos'], 15, self.COLOR_SHIP_GLOW, 100)
                pygame.gfxdraw.aacircle(self.screen, int(ship['pos'][0]), int(ship['pos'][1]), 12, self.COLOR_SHIP_GLOW)
        
        # Draw Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

    def _render_ui(self):
        # Top bar background
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # Fuel Bar
        fuel_ratio = self.player_fuel / self.MAX_FUEL
        bar_width = int(200 * fuel_ratio)
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_FUEL, (10, 10, bar_width, 20))
        fuel_text = self.font_small.render(f"FUEL: {int(self.player_fuel)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (220, 12))
        
        # Turn Counter
        turn_text = self.font_medium.render(f"TURN: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(turn_text, (self.SCREEN_WIDTH - 180, 10))
        
        # Score / Controlled Nebulae
        score_text = self.font_medium.render(f"NEBULAE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 350, 10))
        
        # Selected Ship Info
        ship = self.ships[self.selected_ship_idx]
        ship_info_text = f"SHIP {self.selected_ship_idx + 1} [LVL {ship['level']}]"
        ship_text_render = self.font_medium.render(ship_info_text, True, self.COLOR_SHIP_GLOW)
        self.screen.blit(ship_text_render, (10, self.SCREEN_HEIGHT - 30))

        # Action Feedback
        if self.action_feedback_timer > 0:
            feedback_render = self.font_large.render(self.action_feedback_text, True, self.COLOR_FUEL)
            text_rect = feedback_render.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(feedback_render, text_rect)
            
    # --- Helper Methods ---
    def _generate_stars(self):
        self.stars = []
        for _ in range(self.NUM_STARS):
            self.stars.append({
                'pos': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'size': self.np_random.choice([1, 1, 1, 2])
            })
            
    def _generate_nebulae(self):
        self.nebulae = []
        min_dist = 80
        for i in range(self.NUM_NEBULAE):
            while True:
                pos = (self.np_random.integers(50, self.SCREEN_WIDTH - 50), self.np_random.integers(80, self.SCREEN_HEIGHT - 50))
                if all(np.linalg.norm(np.array(pos) - np.array(n['pos'])) > min_dist for n in self.nebulae):
                    break
            self.nebulae.append({
                'id': i,
                'pos': pos,
                'fuel': self.np_random.integers(self.NEBULA_MIN_FUEL, self.NEBULA_MAX_FUEL),
                'owner': 'neutral',
                'size': self.np_random.integers(10, 20),
                'phase': self.np_random.random() * 2 * math.pi
            })
            
    def _generate_ships(self):
        self.ships = []
        # Ensure there are nebulae to choose from
        if not self.nebulae:
            return
        
        neutral_nebulae = [n for n in self.nebulae if n['owner'] == 'neutral']
        if not neutral_nebulae:
            # If for some reason all nebulae are already owned, pick any
            neutral_nebulae = self.nebulae

        for i in range(self.INITIAL_SHIPS):
            if not neutral_nebulae: break
            start_nebula = self.np_random.choice(neutral_nebulae)
            start_nebula['owner'] = 'player'
            # Remove from list to avoid placing multiple ships at the same start nebula
            neutral_nebulae = [n for n in neutral_nebulae if n['id'] != start_nebula['id']]
            self.score += 1
            self.ships.append({
                'id': i,
                'pos': start_nebula['pos'],
                'level': 1,
                'target_id': start_nebula['id']
            })
            
    def _update_ship_target(self, ship_idx):
        ship = self.ships[ship_idx]
        # If current target is invalid or same as location, find a new one
        if ship['target_id'] == -1 or np.linalg.norm(np.array(ship['pos']) - np.array(self.nebulae[ship['target_id']]['pos'])) < 5:
            closest_dist = float('inf')
            new_target_id = ship['target_id']
            for nebula in self.nebulae:
                if nebula['id'] == ship['target_id']: continue
                dist = np.linalg.norm(np.array(ship['pos']) - np.array(nebula['pos']))
                if dist < closest_dist:
                    closest_dist = dist
                    new_target_id = nebula['id']
            ship['target_id'] = new_target_id

    def _find_next_target(self, direction):
        ship = self.ships[self.selected_ship_idx]
        ship_pos = np.array(ship['pos'])
        
        best_nebula = None
        min_score = float('inf')

        for nebula in self.nebulae:
            if nebula['id'] == ship['target_id']: continue
            
            nebula_pos = np.array(nebula['pos'])
            d_vec = nebula_pos - ship_pos
            
            # Skip nebulae behind the ship relative to direction
            if direction == 1 and d_vec[1] >= 0: continue # Up
            if direction == 2 and d_vec[1] <= 0: continue # Down
            if direction == 3 and d_vec[0] >= 0: continue # Left
            if direction == 4 and d_vec[0] <= 0: continue # Right
            
            # Weight score by alignment with direction
            if direction in [1, 2]: # Up/Down
                score = (d_vec[0]**2) * 2 + abs(d_vec[1]) # Penalize horizontal deviation
            else: # Left/Right
                score = (d_vec[1]**2) * 2 + abs(d_vec[0]) # Penalize vertical deviation
            
            if score < min_score:
                min_score = score
                best_nebula = nebula
        
        if best_nebula:
            ship['target_id'] = best_nebula['id']
            
    def _update_nebulae(self):
        fuel_decay = 0.01 * (self.steps // 100)
        for nebula in self.nebulae:
            regen_amount = self.NEBULA_REGEN_RATE * (1.0 - fuel_decay)
            nebula['fuel'] = min(self.NEBULA_MAX_FUEL, nebula['fuel'] + regen_amount)

    def _create_particle(self, start_pos, color, target_pos):
        self.particles.append({
            'pos': list(start_pos),
            'vel': (np.array(target_pos) - np.array(start_pos)) / 20.0,
            'size': self.np_random.uniform(2, 4),
            'color': color,
            'life': 20
        })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _draw_glowing_circle(self, pos, radius, color, alpha):
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color + (alpha,), (radius, radius), radius)
        self.screen.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the automated test suite.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for manual play
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Nebula Harvest")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop for manual control
    while not terminated and not truncated:
        movement, space, shift = 0, 0, 0
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                action_taken = True # Any key press triggers a step in this turn-based game
        
        if not action_taken:
            # Render the current state even if no action is taken
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30)
            continue

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {info['steps']}, Score: {info['score']}, Fuel: {int(info['fuel'])}, Reward: {reward:.2f}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS
        
    env.close()
    print("Game Over!")