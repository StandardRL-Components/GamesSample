import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:37:50.817908
# Source Brief: brief_00592.md
# Brief Index: 592
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player intercepts incoming missiles.

    The player controls a targeting reticle on a grid and can deploy a limited
    number of EMP (Electromagnetic Pulse) bursts. An EMP burst disables the
    guidance systems of any missiles within its radius, causing them to veer
    off course. The goal is to survive as many waves of missiles as possible.

    **Visuals:**
    - Clean, futuristic neon aesthetic on a dark background.
    - Smooth animations for missiles, trails, and EMP effects.
    - Clear UI for score, wave count, and remaining EMP charges.

    **Gameplay:**
    - Real-time strategy and arcade action.
    - Missiles increase in speed and number over time.
    - Strategic placement of EMPs is key to survival.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Intercept waves of incoming missiles by deploying EMP bursts. Strategically place your "
        "defenses to disable missile guidance systems and survive as long as possible."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the targeting reticle. Press space to deploy an EMP burst."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 8
    GRID_ROWS = 5
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS

    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (30, 60, 90)
    COLOR_TARGET_FILL = (0, 255, 150, 40)
    COLOR_TARGET_OUTLINE = (0, 255, 150)
    COLOR_MISSILE = (255, 50, 50)
    COLOR_MISSILE_TRAIL = (200, 40, 40)
    COLOR_EMP = (0, 255, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (255, 220, 100)

    MAX_STEPS = 2000
    MAX_WAVES = 20
    INITIAL_EMP_CHARGES = 5
    EMP_RADIUS = CELL_WIDTH * 0.8
    INITIAL_MISSILE_SPEED = 1.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 32, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 1
        self.missile_base_speed = self.INITIAL_MISSILE_SPEED
        self.missiles = []
        self.emp_blasts = []
        self.target_cell = (0, 0)
        self.emp_charges = 0
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 1
        self.missile_base_speed = self.INITIAL_MISSILE_SPEED
        
        self.missiles.clear()
        self.emp_blasts.clear()

        self.target_cell = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.emp_charges = self.INITIAL_EMP_CHARGES
        self.prev_space_held = False

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # --- Handle Input and Game Logic ---
        reward += self._handle_input(action)
        self._update_game_state()

        # --- Check for game events ---
        reward += self._check_missile_outcomes()
        
        wave_cleared, wave_reward = self._check_wave_completion()
        if wave_cleared:
            reward += wave_reward

        # --- Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False # Game does not truncate based on time limit in this implementation

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Move Target Reticle ---
        col, row = self.target_cell
        if movement == 1: row -= 1  # Up
        elif movement == 2: row += 1  # Down
        elif movement == 3: col -= 1  # Left
        elif movement == 4: col += 1  # Right
        self.target_cell = (max(0, min(self.GRID_COLS - 1, col)), max(0, min(self.GRID_ROWS - 1, row)))

        # --- Deploy EMP (on rising edge of space press) ---
        if space_held and not self.prev_space_held and self.emp_charges > 0:
            self.emp_charges -= 1
            # Sfx: EMP_FIRE
            return self._deploy_emp()
        
        self.prev_space_held = space_held
        return 0.0

    def _update_game_state(self):
        # Increase missile speed over time
        self.missile_base_speed = self.INITIAL_MISSILE_SPEED + 0.2 * (self.steps // 50)
        
        # Update Missiles
        for m in self.missiles:
            m['pos'][0] += m['vel'][0]
            m['pos'][1] += m['vel'][1]
            m['trail'].append(list(m['pos']))
            if len(m['trail']) > 15:
                m['trail'].pop(0)
        
        # Update EMP blast animations
        self.emp_blasts = [b for b in self.emp_blasts if b['life'] > 0]
        for b in self.emp_blasts:
            b['life'] -= 1

    def _check_missile_outcomes(self):
        reward = 0
        missiles_to_remove = []
        for m in self.missiles:
            # Missile reached the bottom (failure)
            if m['pos'][1] >= self.SCREEN_HEIGHT:
                self.game_over = True
                reward -= 10
                missiles_to_remove.append(m)
                # Sfx: EXPLOSION_LARGE
            # Missile flew off-screen (success if deflected)
            elif not (0 <= m['pos'][0] < self.SCREEN_WIDTH):
                missiles_to_remove.append(m)
        
        self.missiles = [m for m in self.missiles if m not in missiles_to_remove]
        return reward

    def _check_wave_completion(self):
        if not self.missiles and not self.game_over:
            self.current_wave += 1
            self.emp_charges = min(self.INITIAL_EMP_CHARGES + 2, self.emp_charges + 1)
            # Sfx: WAVE_COMPLETE
            
            if self.current_wave > self.MAX_WAVES:
                self.game_over = True
                return True, 100 # Victory bonus
            
            self._spawn_wave()
            return True, 50 # Wave clear bonus
        return False, 0

    def _deploy_emp(self):
        reward = 0
        col, row = self.target_cell
        emp_center = (col * self.CELL_WIDTH + self.CELL_WIDTH / 2, row * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
        
        self.emp_blasts.append({'pos': emp_center, 'life': 20, 'max_life': 20})
        
        for m in self.missiles:
            if not m['deflected']:
                dist = math.hypot(m['pos'][0] - emp_center[0], m['pos'][1] - emp_center[1])
                if dist < self.EMP_RADIUS:
                    m['deflected'] = True
                    self.score += 10
                    reward += 1.0 # Deflection reward
                    # Sfx: MISSILE_DEFLECTED
                    
                    # Veer off course
                    current_speed = math.hypot(m['vel'][0], m['vel'][1])
                    new_vx = random.uniform(-1, 1) * current_speed * 0.7
                    new_vy = math.sqrt(max(0, current_speed**2 - new_vx**2))
                    m['vel'] = [new_vx, new_vy]
        return reward

    def _spawn_wave(self):
        num_missiles = 1 + (self.current_wave - 1) // 3
        spawn_slots = random.sample(range(self.GRID_COLS), k=min(num_missiles, self.GRID_COLS))
        
        for i in range(len(spawn_slots)):
            start_x = spawn_slots[i] * self.CELL_WIDTH + self.CELL_WIDTH / 2
            start_y = -random.uniform(20, 100)
            self.missiles.append({
                'pos': [start_x, start_y],
                'vel': [0, self.missile_base_speed],
                'deflected': False,
                'trail': []
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game(self.screen)
        self._render_ui(self.screen)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, surface):
        # Draw grid
        for i in range(1, self.GRID_COLS):
            pygame.draw.line(surface, self.COLOR_GRID, (i * self.CELL_WIDTH, 0), (i * self.CELL_WIDTH, self.SCREEN_HEIGHT))
        for i in range(1, self.GRID_ROWS):
            pygame.draw.line(surface, self.COLOR_GRID, (0, i * self.CELL_HEIGHT), (self.SCREEN_WIDTH, i * self.CELL_HEIGHT))

        # Draw target reticle
        col, row = self.target_cell
        target_rect = pygame.Rect(col * self.CELL_WIDTH, row * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
        pygame.draw.rect(surface, self.COLOR_TARGET_FILL, target_rect, border_radius=4)
        pygame.draw.rect(surface, self.COLOR_TARGET_OUTLINE, target_rect, 2, border_radius=4)
        
        # Draw missile trails
        for m in self.missiles:
            if len(m['trail']) > 1:
                for i in range(len(m['trail']) - 1):
                    alpha = int(255 * (i / len(m['trail'])))
                    # Using a temp surface for alpha line drawing is complex, so we accept potential visual artifacts
                    # of drawing alpha lines directly. For perfect blending, one would need a separate surface.
                    p1 = (int(m['trail'][i][0]), int(m['trail'][i][1]))
                    p2 = (int(m['trail'][i+1][0]), int(m['trail'][i+1][1]))
                    try:
                        pygame.draw.line(surface, (*self.COLOR_MISSILE_TRAIL, alpha), p1, p2, max(1, int(5 * (i/len(m['trail'])))))
                    except TypeError: # Some pygame versions don't support alpha in color for draw.line
                        pygame.draw.line(surface, self.COLOR_MISSILE_TRAIL, p1, p2, max(1, int(5 * (i/len(m['trail'])))))


        # Draw missiles
        for m in self.missiles:
            pos = (int(m['pos'][0]), int(m['pos'][1]))
            color = self.COLOR_MISSILE if not m['deflected'] else (255, 150, 50)
            self._draw_glow_circle(surface, pos, 6, color)

        # Draw EMP blasts
        for b in self.emp_blasts:
            progress = 1.0 - (b['life'] / b['max_life'])
            current_radius = int(self.EMP_RADIUS * math.sin(progress * math.pi / 2)) # Ease out
            alpha = int(255 * (b['life'] / b['max_life']))
            color = (*self.COLOR_EMP, alpha)
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, current_radius, current_radius, current_radius - 1, color)
            pygame.gfxdraw.aacircle(temp_surf, current_radius, current_radius, current_radius - 2, color)
            surface.blit(temp_surf, (int(b['pos'][0] - current_radius), int(b['pos'][1] - current_radius)))

    def _render_ui(self, surface):
        # Top Left: EMP Charges
        charge_text = f"EMP: {self.emp_charges}"
        self._render_text(surface, charge_text, (10, 10), self.font_ui, self.COLOR_TEXT, align="topleft")
        
        # Top Right: Wave
        wave_text = f"WAVE: {self.current_wave}/{self.MAX_WAVES}"
        self._render_text(surface, wave_text, (self.SCREEN_WIDTH - 10, 10), self.font_ui, self.COLOR_TEXT, align="topright")
        
        # Bottom Center: Score
        score_text = f"SCORE: {self.score}"
        self._render_text(surface, score_text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 35), self.font_score, self.COLOR_SCORE, align="center")

    def _render_text(self, surface, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        surface.blit(text_surface, text_rect)

    def _draw_glow_circle(self, surface, pos, radius, color):
        for i in range(radius, 0, -1):
            alpha = int(255 * (1 - (i / radius))**2)
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), i, glow_color)
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), radius, color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "emp_charges": self.emp_charges}

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    # Un-comment the line below to run with display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Manual play loop
    manual_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Missile Intercept")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]
    
    print("\n--- Manual Control ---")
    print("Arrows: Move Target")
    print("Space: Deploy EMP")
    print("Q: Quit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        
        # Movement (mutually exclusive)
        action[0] = 0 # No movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        # Actions (can be simultaneous)
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0.1: # Print non-trivial rewards
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Render the observation to the manual display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        manual_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}, Total Steps: {info['steps']}")
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS

    env.close()