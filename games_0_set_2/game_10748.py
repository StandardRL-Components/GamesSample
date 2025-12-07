import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:52:32.209018
# Source Brief: brief_00748.md
# Brief Index: 748
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for "Dreamscape Raider".

    The agent explores a procedurally generated dreamscape, collecting dream
    fragments while avoiding nightmares. The agent can move, use a special
    ability (teleport), and rewind time. The goal is to maximize the score
    by collecting fragments within a time limit.

    **Visuals:**
    - Player: Glowing yellow orb.
    - Dream Fragments: Pulsing cyan orbs.
    - Nightmares: Pulsating red/purple jagged shapes.
    - Background: Dark starfield with a faint grid.

    **Gameplay:**
    - Collect fragments for points.
    - Avoid nightmares, which penalize the score and can end the game.
    - Every 100 steps, the dreamscape resets with more fragments and potentially
      more nightmares, increasing difficulty.
    - Collecting 50 fragments unlocks the Teleport ability.
    - The agent has a limited number of time rewinds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Explore a procedurally generated dreamscape, collecting dream fragments while avoiding nightmares. "
        "Use teleport and time-rewind abilities to maximize your score."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to teleport (when unlocked) and shift to rewind time."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = 20
        self.GAME_AREA_X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GAME_AREA_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_STEPS = 1000
        self.MAX_NIGHTMARE_COLLISIONS = 5
        self.HISTORY_BUFFER_SIZE = 50

        # --- Colors ---
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_GRID = (30, 20, 50, 100)
        self.COLOR_PLAYER = (255, 220, 50)
        self.COLOR_FRAGMENT = (0, 255, 200)
        self.COLOR_NIGHTMARE = (180, 20, 80)
        self.COLOR_NIGHTMARE_PULSE = (255, 50, 120)
        self.COLOR_TEXT = (230, 230, 255)
        self.COLOR_BAR = (40, 40, 80)
        self.COLOR_BAR_FILL = (0, 255, 200)
        self.COLOR_TELEPORT_ICON = (100, 150, 255)
        self.COLOR_REWIND_ICON = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.fragments = []
        self.nightmares = []
        self.particles = []
        self.state_history = deque(maxlen=self.HISTORY_BUFFER_SIZE)
        self.rewind_charges = 0
        self.nightmare_collisions = 0
        self.total_fragments_collected = 0
        self.teleport_unlocked = False
        self.score_milestones_reached = set()
        self.last_space_held = False
        self.last_shift_held = False

        # --- Pre-generate background elements for performance ---
        self.static_background = self._create_static_background()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.nightmare_collisions = 0
        self.total_fragments_collected = 0
        self.teleport_unlocked = False
        self.score_milestones_reached = set()
        self.rewind_charges = 3

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.player_visual_pos = self._grid_to_pixel(self.player_pos)
        
        self.last_space_held = False
        self.last_shift_held = False

        self.fragments.clear()
        self.nightmares.clear()
        self.particles.clear()
        self.state_history.clear()

        self._cycle_dreamscape(initial=True)
        self._save_state()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        # Action: Rewind Time
        if shift_pressed and self.rewind_charges > 0 and len(self.state_history) > 1:
            self.rewind_charges -= 1
            self._load_state(self.state_history.pop())
            # Sfx: Rewind sound
            self._add_particles(self.player_visual_pos, (255, 100, 100), 20, 2.0)
        else:
            # --- Update Game State (if not rewinding) ---
            
            # Action: Movement
            if movement > 0:
                target_pos = list(self.player_pos)
                if movement == 1: target_pos[1] -= 1  # Up
                elif movement == 2: target_pos[1] += 1  # Down
                elif movement == 3: target_pos[0] -= 1  # Left
                elif movement == 4: target_pos[0] += 1  # Right
                
                if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT:
                    self.player_pos = target_pos
                    # Sfx: Player move
            
            # Action: Use Skill (Teleport)
            elif space_pressed and self.teleport_unlocked:
                self._use_teleport()
                # Sfx: Teleport sound
            
            # Update Nightmares
            for nightmare in self.nightmares:
                self._move_nightmare(nightmare)

            # --- Calculate Rewards & Check Collisions ---
            reward += self._calculate_reward()

            # --- Periodic Difficulty Increase ---
            if self.steps > 0 and self.steps % 100 == 0:
                self._cycle_dreamscape()
                
            self._save_state()

        # --- Update Visuals ---
        self._update_visuals()

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = False
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _calculate_reward(self):
        reward = 0.0

        # Fragment collection
        if tuple(self.player_pos) in self.fragments:
            self.fragments.remove(tuple(self.player_pos))
            self.score += 1
            self.total_fragments_collected += 1
            reward += 1.0
            # Sfx: Fragment collect
            self._add_particles(self.player_visual_pos, self.COLOR_FRAGMENT, 30, 3.0)

            # Check for skill unlock
            if not self.teleport_unlocked and self.total_fragments_collected >= 50:
                self.teleport_unlocked = True
                reward += 5.0 # Event-based reward for unlocking skill

        # Nightmare collision
        for nightmare in self.nightmares:
            if self.player_pos == nightmare['pos']:
                self.score -= 10
                reward -= 10.0
                self.nightmare_collisions += 1
                # Sfx: Player hit
                self._add_particles(self.player_visual_pos, self.COLOR_NIGHTMARE, 40, 4.0)
                break # only one collision per step

        # Proximity penalty
        adjacent_nightmares = 0
        for nightmare in self.nightmares:
            dist = abs(self.player_pos[0] - nightmare['pos'][0]) + abs(self.player_pos[1] - nightmare['pos'][1])
            if dist == 1:
                adjacent_nightmares += 1
        reward -= 0.1 * adjacent_nightmares

        # Score milestone rewards
        if self.score >= 100 and 100 not in self.score_milestones_reached:
            reward += 25.0
            self.score_milestones_reached.add(100)
        if self.score >= 200 and 200 not in self.score_milestones_reached:
            reward += 50.0
            self.score_milestones_reached.add(200)

        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self.nightmare_collisions >= self.MAX_NIGHTMARE_COLLISIONS:
            return True
        return False

    def _cycle_dreamscape(self, initial=False):
        self.fragments.clear()
        
        # Increase nightmare count based on total fragments collected, not steps
        # Brief: "number of nightmares increases by 1 every 200 steps"
        # Let's use steps for simplicity as per brief.
        num_nightmares = 1 + self.steps // 200
        if initial:
            num_nightmares = 1
        
        # Brief: "Dream fragment distribution changes every 100 steps, becoming denser"
        num_fragments = 15 + (self.steps // 100) * 5
        
        occupied_tiles = {tuple(self.player_pos)}
        
        # Respawn nightmares
        if len(self.nightmares) < num_nightmares:
            for _ in range(num_nightmares - len(self.nightmares)):
                pos = self._get_random_empty_tile(occupied_tiles)
                occupied_tiles.add(tuple(pos))
                self.nightmares.append({
                    'pos': pos,
                    'visual_pos': self._grid_to_pixel(pos),
                    'pattern': random.choice(['horizontal', 'vertical']),
                    'dir': random.choice([-1, 1])
                })

        # Respawn fragments
        for _ in range(num_fragments):
            pos = self._get_random_empty_tile(occupied_tiles)
            occupied_tiles.add(tuple(pos))
            self.fragments.append(tuple(pos))
        # Sfx: Dreamscape cycle sound

    def _use_teleport(self):
        self._add_particles(self.player_visual_pos, self.COLOR_TELEPORT_ICON, 50, 5.0)
        occupied = {tuple(n['pos']) for n in self.nightmares}
        self.player_pos = self._get_random_empty_tile(occupied)
        # Snap visual position immediately after teleport
        self.player_visual_pos = self._grid_to_pixel(self.player_pos)

    def _get_random_empty_tile(self, occupied_tiles):
        while True:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in occupied_tiles:
                return list(pos)

    def _save_state(self):
        state = {
            'score': self.score,
            'player_pos': list(self.player_pos),
            'fragments': set(self.fragments),
            'nightmares': [n.copy() for n in self.nightmares],
            'nightmare_collisions': self.nightmare_collisions,
            'total_fragments_collected': self.total_fragments_collected,
            'teleport_unlocked': self.teleport_unlocked,
            'score_milestones_reached': self.score_milestones_reached.copy()
        }
        self.state_history.append(state)

    def _load_state(self, state):
        self.score = state['score']
        self.player_pos = state['player_pos']
        self.fragments = list(state['fragments'])
        self.nightmares = state['nightmares']
        self.nightmare_collisions = state['nightmare_collisions']
        self.total_fragments_collected = state['total_fragments_collected']
        self.teleport_unlocked = state['teleport_unlocked']
        self.score_milestones_reached = state['score_milestones_reached']

    def _move_nightmare(self, nightmare):
        pos = nightmare['pos']
        if nightmare['pattern'] == 'horizontal':
            pos[0] += nightmare['dir']
            if not (0 <= pos[0] < self.GRID_WIDTH):
                nightmare['dir'] *= -1
                pos[0] += 2 * nightmare['dir']
        else: # vertical
            pos[1] += nightmare['dir']
            if not (0 <= pos[1] < self.GRID_HEIGHT):
                nightmare['dir'] *= -1
                pos[1] += 2 * nightmare['dir']
        # Ensure position is valid after bounce
        pos[0] = np.clip(pos[0], 0, self.GRID_WIDTH - 1)
        pos[1] = np.clip(pos[1], 0, self.GRID_HEIGHT - 1)

    def _update_visuals(self, lerp_rate=0.25):
        # Interpolate player position
        target_pixel_pos = self._grid_to_pixel(self.player_pos)
        self.player_visual_pos[0] = self.player_visual_pos[0] * (1 - lerp_rate) + target_pixel_pos[0] * lerp_rate
        self.player_visual_pos[1] = self.player_visual_pos[1] * (1 - lerp_rate) + target_pixel_pos[1] * lerp_rate

        # Interpolate nightmare positions
        for n in self.nightmares:
            target_pixel_pos = self._grid_to_pixel(n['pos'])
            n['visual_pos'][0] = n['visual_pos'][0] * (1-lerp_rate) + target_pixel_pos[0] * lerp_rate
            n['visual_pos'][1] = n['visual_pos'][1] * (1-lerp_rate) + target_pixel_pos[1] * lerp_rate
            
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _grid_to_pixel(self, grid_pos):
        x = self.GAME_AREA_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GAME_AREA_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return [x, y]

    def _get_observation(self):
        self.screen.blit(self.static_background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rewind_charges": self.rewind_charges,
            "teleport_unlocked": self.teleport_unlocked
        }

    def _create_static_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        bg.fill(self.COLOR_BG)
        # Draw stars
        for _ in range(150):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.randint(1, 2)
            brightness = random.randint(50, 150)
            pygame.draw.rect(bg, (brightness, brightness, brightness), (x, y, size, size))
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.GAME_AREA_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(bg, self.COLOR_GRID, (px, self.GAME_AREA_Y_OFFSET), (px, self.HEIGHT - self.GAME_AREA_Y_OFFSET))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GAME_AREA_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(bg, self.COLOR_GRID, (self.GAME_AREA_X_OFFSET, py), (self.WIDTH - self.GAME_AREA_X_OFFSET, py))
        return bg

    def _render_game(self):
        # Render particles (underneath other elements)
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(max(1, p['size'] * (p['life'] / p['max_life'])))
            self._draw_glow_circle(self.screen, p['pos'], size, color, 0.5)

        # Render fragments
        for frag_pos in self.fragments:
            pixel_pos = self._grid_to_pixel(frag_pos)
            pulse = (math.sin(self.steps * 0.1 + frag_pos[0]) + 1) / 2
            radius = int(self.CELL_SIZE * 0.2 + pulse * 3)
            self._draw_glow_circle(self.screen, pixel_pos, radius, self.COLOR_FRAGMENT)

        # Render nightmares
        for nightmare in self.nightmares:
            self._draw_pulsing_nightmare(self.screen, nightmare['visual_pos'])

        # Render player
        self._draw_glow_circle(self.screen, self.player_visual_pos, int(self.CELL_SIZE * 0.4), self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Time remaining bar
        bar_width = 200
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 15
        bar_y = 15
        progress = self.steps / self.MAX_STEPS
        pygame.draw.rect(self.screen, self.COLOR_BAR, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_BAR_FILL, (bar_x, bar_y, bar_width * progress, bar_height), border_radius=4)
        
        # Skills and Charges UI
        ui_y = self.HEIGHT - 30
        
        # Rewind charges
        rewind_text = self.font_small.render(f"REWIND", True, self.COLOR_REWIND_ICON)
        self.screen.blit(rewind_text, (15, ui_y))
        for i in range(self.rewind_charges):
            self._draw_glow_circle(self.screen, [80 + i * 15, ui_y + 8], 4, self.COLOR_REWIND_ICON, 0.7)

        # Teleport status
        teleport_color = self.COLOR_TELEPORT_ICON if self.teleport_unlocked else (50, 60, 80)
        teleport_text = self.font_small.render(f"TELEPORT", True, teleport_color)
        self.screen.blit(teleport_text, (150, ui_y))
        status_text = "READY" if self.teleport_unlocked else "LOCKED"
        status_surf = self.font_small.render(status_text, True, teleport_color)
        self.screen.blit(status_surf, (225, ui_y))

    def _draw_glow_circle(self, surface, pos, radius, color, glow_factor=2.0):
        x, y = int(pos[0]), int(pos[1])
        # Draw glow layers
        for i in range(int(radius * glow_factor), radius, -2):
            alpha = 50 * (1 - (i - radius) / (radius * (glow_factor - 1)))
            pygame.gfxdraw.filled_circle(surface, x, y, i, (*color, alpha))
        # Draw core circle
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, color)

    def _draw_pulsing_nightmare(self, surface, pos):
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        main_color = tuple(int(c1 * (1 - pulse) + c2 * pulse) for c1, c2 in zip(self.COLOR_NIGHTMARE, self.COLOR_NIGHTMARE_PULSE))
        
        size = self.CELL_SIZE * 0.4 + pulse * 4
        points = []
        for i in range(5):
            angle = 2 * math.pi * i / 5 + self.steps * 0.05
            r = size * (0.8 if i % 2 == 0 else 1.2)
            points.append((int(pos[0] + r * math.cos(angle)), int(pos[1] + r * math.sin(angle))))
        
        pygame.gfxdraw.aapolygon(surface, points, main_color)
        pygame.gfxdraw.filled_polygon(surface, points, main_color)

    def _add_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.uniform(2, 5)
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # The following code will not run in a headless environment
    # It is for local testing with a display.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dreamscape Raider")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R:      Reset Environment")
    print("Q:      Quit")
    print("----------------\n")

    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
                if event.key == pygame.K_q:
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

        # Render the observation to the display window
        # Pygame uses (width, height), so we need to transpose back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Limit to 30 FPS for smooth visuals

    env.close()