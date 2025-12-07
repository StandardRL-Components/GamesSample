import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:59:02.392648
# Source Brief: brief_00135.md
# Brief Index: 135
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
        "Defend a central cell by synthesizing matching proteins to neutralize incoming waves of pathogens."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to select a protein. Press space to synthesize more of the selected protein. Press shift to deploy defenses."
    )
    auto_advance = False

    # --- VISUAL AND GAMEPLAY CONSTANTS ---
    
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_CELL_WALL = (30, 80, 120)
    COLOR_CELL_GLOW = (40, 100, 150)
    
    PROTEIN_TYPES = [
        {'name': 'Icosa-P', 'color': (0, 200, 255), 'pathogen_shape': 'triangle'},
        {'name': 'Recta-G', 'color': (255, 200, 0), 'pathogen_shape': 'square'},
        {'name': 'Hexa-V', 'color': (200, 0, 255), 'pathogen_shape': 'hexagon'},
    ]
    
    COLOR_PATHOGEN_BASE = (255, 50, 50)
    COLOR_DEFENSE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    COLOR_UI_BG = (20, 40, 60, 180)
    COLOR_UI_HIGHLIGHT = (255, 255, 100)
    COLOR_HEALTH_HIGH = (80, 220, 80)
    COLOR_HEALTH_MID = (240, 200, 60)
    COLOR_HEALTH_LOW = (220, 50, 50)
    
    # Game Parameters
    MAX_STEPS = 1000
    TOTAL_WAVES = 20
    INITIAL_CELL_HEALTH = 100
    CELL_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    CELL_RADIUS = 180
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 30, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)

        # Action processing state
        self.prev_action = [0, 0, 0]

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cell_health = 0
        self.current_wave = 0
        self.pathogens = []
        self.protein_counts = []
        self.synthesis_rate = 0
        self.unlocked_protein_types = 0
        self.selected_protein_index = 0
        self.particles = []
        self.screen_flash = 0
        
        # Initialize state
        # self.reset() # reset is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cell_health = self.INITIAL_CELL_HEALTH
        self.current_wave = 0 # Will be incremented to 1
        
        self.pathogens = []
        self.particles = []
        
        self.unlocked_protein_types = 1
        self.protein_counts = [10] * len(self.PROTEIN_TYPES)
        self.synthesis_rate = 1.0
        self.selected_protein_index = 0
        
        self.prev_action = [0, 0, 0]
        self.screen_flash = 0

        self._spawn_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        turn_taken = False

        # --- Process Actions (convert held to pressed) ---
        move_action, space_pressed, shift_pressed = self._process_raw_action(action)

        # --- Handle Selection (free action) ---
        if move_action == 1: # Up
            self.selected_protein_index = (self.selected_protein_index - 1) % self.unlocked_protein_types
        elif move_action == 2: # Down
            self.selected_protein_index = (self.selected_protein_index + 1) % self.unlocked_protein_types

        # --- Handle Clone Action ---
        if space_pressed:
            # // SFX: clone_success
            clone_amount = int(5 * self.synthesis_rate)
            self.protein_counts[self.selected_protein_index] += clone_amount
            turn_taken = True

        # --- Handle Match & Deploy Action ---
        if shift_pressed:
            matched_any = False
            for i in range(self.unlocked_protein_types):
                if self.protein_counts[i] >= 2:
                    # // SFX: deploy_defense
                    self.protein_counts[i] -= 2
                    self._create_defense_wave(i)
                    matched_any = True
            if matched_any:
                turn_taken = True
        
        # --- Update Game World if a Turn was Taken ---
        if turn_taken:
            self.steps += 1
            reward += self._update_pathogens()

        # --- Update Continuous Systems (Particles) ---
        reward += self._update_particles()
        
        # --- Handle Wave Completion ---
        if turn_taken and not self.pathogens and self.current_wave <= self.TOTAL_WAVES:
            reward += 5 # Wave complete bonus
            self._spawn_next_wave()
        
        # --- Update Screen Effects ---
        if self.screen_flash > 0:
            self.screen_flash -= 1
            
        # --- Check for Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.cell_health <= 0:
                reward = -100 # Loss penalty
            elif self.current_wave > self.TOTAL_WAVES:
                reward = 100 # Victory bonus
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _process_raw_action(self, action):
        move_action = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # Detect rising edge for presses
        space_pressed = space_held and not self.prev_action[1]
        shift_pressed = shift_held and not self.prev_action[2]
        
        # Debounce movement for selection
        move_pressed = move_action != 0 and self.prev_action[0] == 0
        
        self.prev_action = [move_action, space_held, shift_held]
        
        return move_action if move_pressed else 0, space_pressed, shift_pressed

    def _spawn_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            return

        # Unlock new types and increase synthesis rate
        if self.current_wave % 5 == 0 and self.unlocked_protein_types < len(self.PROTEIN_TYPES):
            self.unlocked_protein_types += 1
        if self.current_wave % 10 == 0:
            self.synthesis_rate += 0.1

        num_pathogens = 3 + (self.current_wave - 1)
        speed = 0.5 + (self.current_wave // 2) * 0.05
        
        for _ in range(num_pathogens):
            angle = self.np_random.uniform(0, 2 * math.pi)
            spawn_dist = self.CELL_RADIUS + self.np_random.uniform(20, 50)
            pos = [
                self.CELL_CENTER[0] + spawn_dist * math.cos(angle),
                self.CELL_CENTER[1] + spawn_dist * math.sin(angle)
            ]
            
            # Pathogen moves towards the center
            direction_to_center = np.array(self.CELL_CENTER) - np.array(pos)
            norm = np.linalg.norm(direction_to_center)
            if norm == 0: continue
            velocity = (direction_to_center / norm) * speed

            pathogen_type = self.np_random.integers(0, self.unlocked_protein_types)
            
            self.pathogens.append({
                'pos': pos,
                'vel': velocity,
                'type': pathogen_type,
                'size': 10,
                'angle': self.np_random.uniform(0, 2 * math.pi)
            })

    def _update_pathogens(self):
        reward = 0
        for p in self.pathogens:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            
            dist_from_center = math.hypot(p['pos'][0] - self.CELL_CENTER[0], p['pos'][1] - self.CELL_CENTER[1])
            if dist_from_center <= self.CELL_RADIUS:
                # // SFX: cell_damage
                self.cell_health = max(0, self.cell_health - 10)
                reward -= 0.1 # Penalty for reaching membrane
                self.screen_flash = 5 # Flash red for 5 frames
                p['health'] = 0 # Mark for removal
        
        self.pathogens = [p for p in self.pathogens if p.get('health', 1) > 0]
        return reward

    def _create_defense_wave(self, protein_type):
        for i in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            self.particles.append({
                'pos': list(self.CELL_CENTER),
                'vel': [speed * math.cos(angle), speed * math.sin(angle)],
                'type': 'defense',
                'protein_type': protein_type,
                'lifespan': 60,
                'color': self.COLOR_DEFENSE
            })

    def _update_particles(self):
        reward = 0
        for part in self.particles:
            part['pos'][0] += part['vel'][0]
            part['pos'][1] += part['vel'][1]
            part['lifespan'] -= 1
            
            if part['type'] == 'defense':
                for p in self.pathogens:
                    if p['type'] == part['protein_type']:
                        if math.hypot(part['pos'][0] - p['pos'][0], part['pos'][1] - p['pos'][1]) < p['size']:
                            # // SFX: pathogen_destroy
                            p['health'] = 0 # Mark pathogen for removal
                            part['lifespan'] = 0 # Mark particle for removal
                            reward += 1 # Reward for destroying pathogen
                            # Create explosion effect
                            for _ in range(15):
                                angle = self.np_random.uniform(0, 2 * math.pi)
                                speed = self.np_random.uniform(1, 3)
                                self.particles.append({
                                    'pos': list(p['pos']),
                                    'vel': [speed * math.cos(angle), speed * math.sin(angle)],
                                    'type': 'explosion',
                                    'lifespan': self.np_random.integers(10, 20),
                                    'color': self.PROTEIN_TYPES[p['type']]['color']
                                })
                            break # Particle can only hit one pathogen
        
        self.pathogens = [p for p in self.pathogens if p.get('health', 1) > 0]
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        return reward

    def _check_termination(self):
        return self.cell_health <= 0 or self.current_wave > self.TOTAL_WAVES

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "health": self.cell_health}

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_and_cell()
        self._render_particles()
        self._render_pathogens()
        
        if self.screen_flash > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = 100 * (self.screen_flash / 5)
            flash_surface.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surface, (0, 0))

        self._render_ui()

        if self.game_over:
            self._render_game_over()

    def _render_background_and_cell(self):
        # Cell glow
        for i in range(15):
            alpha = 40 * (1 - i / 15)
            pygame.gfxdraw.aacircle(self.screen, self.CELL_CENTER[0], self.CELL_CENTER[1], self.CELL_RADIUS + i, (*self.COLOR_CELL_GLOW, int(alpha)))
        
        # Cell wall
        pygame.gfxdraw.aacircle(self.screen, self.CELL_CENTER[0], self.CELL_CENTER[1], self.CELL_RADIUS, self.COLOR_CELL_WALL)
        pygame.gfxdraw.filled_circle(self.screen, self.CELL_CENTER[0], self.CELL_CENTER[1], self.CELL_RADIUS, self.COLOR_CELL_WALL)
        
        # Inner dark area
        pygame.gfxdraw.filled_circle(self.screen, self.CELL_CENTER[0], self.CELL_CENTER[1], self.CELL_RADIUS - 2, self.COLOR_BG)

    def _render_pathogens(self):
        for p in self.pathogens:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            size = p['size'] + 2 * math.sin(pygame.time.get_ticks() * 0.005 + p['angle'])
            color = self.PROTEIN_TYPES[p['type']]['color']
            shape = self.PROTEIN_TYPES[p['type']]['pathogen_shape']
            
            points = []
            if shape == 'triangle':
                for i in range(3):
                    angle = p['angle'] + (i * 2 * math.pi / 3)
                    points.append((x + size * math.cos(angle), y + size * math.sin(angle)))
            elif shape == 'square':
                for i in range(4):
                    angle = p['angle'] + (i * 2 * math.pi / 4) + math.pi/4
                    points.append((x + size * math.cos(angle), y + size * math.sin(angle)))
            elif shape == 'hexagon':
                for i in range(6):
                    angle = p['angle'] + (i * 2 * math.pi / 6)
                    points.append((x + size * math.cos(angle), y + size * math.sin(angle)))
            
            if points:
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = 255
            if p['lifespan'] < 20:
                alpha = int(255 * (p['lifespan'] / 20))
            
            color_val = p['color']
            if len(color_val) == 3:
                color = (*color_val, alpha)
            else:
                color = (color_val[0], color_val[1], color_val[2], alpha)
            
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            
            if p['type'] == 'defense':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)
            elif p['type'] == 'explosion':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

    def _render_ui(self):
        # --- Score and Wave ---
        self._render_text(f"SCORE: {int(self.score)}", (10, 10), self.font_small, self.COLOR_TEXT)
        wave_text = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}" if self.current_wave <= self.TOTAL_WAVES else "VICTORY!"
        self._render_text(wave_text, (self.SCREEN_WIDTH - 10, 10), self.font_small, self.COLOR_TEXT, align="topright")

        # --- Health Bar ---
        bar_width, bar_height = 200, 20
        bar_x, bar_y = (self.SCREEN_WIDTH - bar_width) // 2, 10
        health_ratio = self.cell_health / self.INITIAL_CELL_HEALTH if self.INITIAL_CELL_HEALTH > 0 else 0
        
        health_color = self.COLOR_HEALTH_LOW
        if health_ratio > 0.66: health_color = self.COLOR_HEALTH_HIGH
        elif health_ratio > 0.33: health_color = self.COLOR_HEALTH_MID
        
        pygame.draw.rect(self.screen, (0,0,0), (bar_x-2, bar_y-2, bar_width+4, bar_height+4))
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
        self._render_text("CELL HEALTH", (self.SCREEN_WIDTH // 2, bar_y + bar_height + 8), self.font_small, self.COLOR_TEXT, align="midtop")

        # --- Protein/Cloning UI ---
        ui_width, ui_height = 180, 120
        ui_x, ui_y = 10, self.SCREEN_HEIGHT - ui_height - 10
        
        s = pygame.Surface((ui_width, ui_height), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (ui_x, ui_y))
        pygame.draw.rect(self.screen, self.COLOR_CELL_GLOW, (ui_x, ui_y, ui_width, ui_height), 1)
        
        self._render_text("PROTEIN SYNTHESIS", (ui_x + ui_width/2, ui_y + 10), self.font_small, self.COLOR_TEXT, align="midtop")
        
        for i in range(self.unlocked_protein_types):
            y_offset = ui_y + 40 + i * 25
            is_selected = i == self.selected_protein_index
            
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_UI_HIGHLIGHT, (ui_x+5, y_offset-3, ui_width-10, 22), 1)
            
            p_type = self.PROTEIN_TYPES[i]
            pygame.draw.rect(self.screen, p_type['color'], (ui_x + 15, y_offset, 15, 15))
            
            text = f"{p_type['name']}: {self.protein_counts[i]}"
            self._render_text(text, (ui_x + 40, y_offset), self.font_small, self.COLOR_TEXT)

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        message = "VICTORY" if self.current_wave > self.TOTAL_WAVES else "CELL COMPROMISED"
        color = self.COLOR_HEALTH_HIGH if self.current_wave > self.TOTAL_WAVES else self.COLOR_HEALTH_LOW
        
        self._render_text(message, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 30), self.font_huge, color, align="center")
        self._render_text(f"Final Score: {int(self.score)}", (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 40), self.font_large, self.COLOR_TEXT, align="center")

    def _render_text(self, text, pos, font, color, align="topleft"):
        shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        main_text = font.render(text, True, color)
        
        rect = main_text.get_rect()
        if align == "topleft": rect.topleft = pos
        elif align == "topright": rect.topright = pos
        elif align == "center": rect.center = pos
        elif align == "midtop": rect.midtop = pos
            
        self.screen.blit(shadow, rect.move(1, 1))
        self.screen.blit(main_text, rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not be run by the autograder
    render_mode = "human" # "human" or "rgb_array"
    
    if render_mode == "human":
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.display.init()
        pygame.font.init()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    if render_mode == "human":
        pygame.display.set_caption("Cell Defense")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Human Input to Action Mapping ---
        move = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space = 0 # 0=released, 1=held
        shift = 0 # 0=released, 1=held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: move = 1
        elif keys[pygame.K_DOWN]: move = 2
        elif keys[pygame.K_LEFT]: move = 3
        elif keys[pygame.K_RIGHT]: move = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [move, space, shift]

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Game Over! Final Score: {total_reward}")
                print(info)
        
        # --- Rendering ---
        if render_mode == "human":
            # The observation is already the rendered screen, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            env.clock.tick(30) # Run at 30 FPS

    env.close()