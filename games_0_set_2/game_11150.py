import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:40:52.563545
# Source Brief: brief_01150.md
# Brief Index: 1150
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A 2D platformer where you collect letters to spell biological terms. "
        "Activate organelles to unlock new areas and complete the level."
    )
    user_guide = (
        "Use ←→ to move and space to jump. Stand near an organelle and press shift "
        "to spell a word with your collected letters."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4.5
    GRAVITY = 0.4
    JUMP_STRENGTH = -9.5
    PARTICLE_LIFESPAN = 30
    ACTIVATION_RANGE = 50

    # --- Colors ---
    COLOR_BG_DARK = (15, 20, 45)
    COLOR_BG_LIGHT = (30, 40, 80)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_PLATFORM = (100, 120, 180)
    COLOR_PLATFORM_ACTIVE = (200, 220, 255)
    COLOR_LETTER = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (40, 50, 100, 150)
    COLOR_SUCCESS = (0, 255, 128)
    COLOR_FAIL = (255, 50, 50)
    
    ORGANELLE_COLORS = {
        "MITOCHONDRION": ((255, 80, 80), (150, 40, 40)), # Active, Inactive
        "RIBOSOME": ((80, 120, 255), (40, 60, 150)),
        "NUCLEUS": ((200, 80, 255), (100, 40, 150)),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = False
        self.inventory = None
        self.platforms = None
        self.letters = None
        self.organelles = None
        self.particles = None
        self.bg_particles = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # self.reset() # reset is called by the wrapper/runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = np.array([50.0, 350.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_grounded = False
        
        self.inventory = []
        self.particles = []
        
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        self._initialize_level()

        if self.bg_particles is None:
            self.bg_particles = [
                {
                    'pos': np.array([random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)]),
                    'radius': random.uniform(1, 4),
                    'speed': random.uniform(0.1, 0.3)
                } for _ in range(50)
            ]
        
        return self._get_observation(), self._get_info()

    def _initialize_level(self):
        self.platforms = [
            # Floor
            {'rect': pygame.Rect(0, 380, 640, 20), 'type': 'static', 'active': True},
            # Mid-level static platform
            {'rect': pygame.Rect(80, 230, 100, 20), 'type': 'static', 'active': True},
            # High static platform
            {'rect': pygame.Rect(500, 120, 140, 20), 'type': 'static', 'active': True},
            # Dynamic platforms (initially inactive)
            {'rect': pygame.Rect(400, 280, 100, 20), 'type': 'dynamic', 'active': False, 'id': 1},
            {'rect': pygame.Rect(250, 150, 100, 20), 'type': 'dynamic', 'active': False, 'id': 2},
        ]

        self.letters = [
            {'char': 'C', 'pos': np.array([150.0, 355.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'E', 'pos': np.array([250.0, 355.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'L', 'pos': np.array([350.0, 355.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'L', 'pos': np.array([450.0, 355.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},

            {'char': 'G', 'pos': np.array([425.0, 255.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'E', 'pos': np.array([475.0, 255.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'N', 'pos': np.array([105.0, 205.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'E', 'pos': np.array([155.0, 205.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},

            {'char': 'P', 'pos': np.array([275.0, 125.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'R', 'pos': np.array([325.0, 125.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'O', 'pos': np.array([375.0, 125.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
            {'char': 'T', 'pos': np.array([425.0, 125.0]), 'collected': False, 'bob_offset': random.uniform(0, math.pi * 2)},
        ]

        self.organelles = [
            {'name': 'MITOCHONDRION', 'pos': np.array([580.0, 355.0]), 'word': 'CELL', 'activated': False, 'activates_id': 1},
            {'name': 'RIBOSOME', 'pos': np.array([50.0, 205.0]), 'word': 'GENE', 'activated': False, 'activates_id': 2},
            {'name': 'NUCLEUS', 'pos': np.array([570.0, 95.0]), 'word': 'PROTEIN', 'activated': False, 'activates_id': -1}, # -1 is win condition
        ]

    def step(self, action):
        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        # Horizontal Movement
        if movement == 3: # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:
            self.player_vel[0] *= 0.8 # Friction

        # Jump (on rising edge)
        if space_held and not self.prev_space_held and self.is_grounded:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_grounded = False
            self._create_particles(self.player_pos + np.array([0, self.PLAYER_SIZE]), 10, self.COLOR_PLAYER, angle_range=(math.pi/4, 3*math.pi/4)) # Jump dust
            # sfx: jump

        # Spell (on rising edge)
        if shift_held and not self.prev_shift_held:
            spell_reward = self._handle_spelling()
            reward += spell_reward
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Physics & State ---
        self.player_vel[1] += self.GRAVITY
        self.player_pos += self.player_vel
        self.is_grounded = False

        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        if self.player_pos[0] in [self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE]:
            self.player_vel[0] = 0

        # Collision with platforms
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE * 2)
        for plat in self.platforms:
            if plat['active'] and player_rect.colliderect(plat['rect']):
                # Player was moving down and is now overlapping
                if self.player_vel[1] > 0 and player_rect.bottom > plat['rect'].top and player_rect.centery < plat['rect'].centery:
                    self.player_pos[1] = plat['rect'].top
                    self.player_vel[1] = 0
                    if not self.is_grounded:
                        self._create_particles(self.player_pos.copy(), 5, self.COLOR_PLATFORM, angle_range=(math.pi/4, 3*math.pi/4)) # Land dust
                        # sfx: land
                    self.is_grounded = True
                # Player was moving up
                elif self.player_vel[1] < 0 and player_rect.top < plat['rect'].bottom:
                    self.player_pos[1] = plat['rect'].bottom + self.PLAYER_SIZE
                    self.player_vel[1] = 0
                # Horizontal collision
                else:
                    self.player_vel[0] = 0

        # Letter collection
        for letter in self.letters:
            if not letter['collected'] and np.linalg.norm(self.player_pos - letter['pos']) < self.PLAYER_SIZE + 10:
                letter['collected'] = True
                self.inventory.append(letter['char'])
                reward += 0.1
                self.score += 10
                self._create_particles(letter['pos'], 15, self.COLOR_LETTER)
                # sfx: collect_letter

        # Update particles
        self._update_particles()

        # Update bobbing letters and background
        for letter in self.letters:
            letter['bob_offset'] += 0.05
        for p in self.bg_particles:
            p['pos'][1] += p['speed']
            if p['pos'][1] > self.SCREEN_HEIGHT + p['radius']:
                p['pos'][1] = -p['radius']
                p['pos'][0] = random.uniform(0, self.SCREEN_WIDTH)

        # --- Termination Conditions ---
        self.steps += 1
        terminated = False
        truncated = False
        if self.player_pos[1] > self.SCREEN_HEIGHT: # Fell off
            reward -= 10
            self.score -= 1000
            terminated = True
            self.game_over = True
        elif self.win:
            reward += 100
            self.score += 10000
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_spelling(self):
        closest_organelle = None
        min_dist = float('inf')
        
        for org in self.organelles:
            if not org['activated']:
                dist = np.linalg.norm(self.player_pos - org['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_organelle = org
        
        if closest_organelle and min_dist < self.ACTIVATION_RANGE:
            inventory_counts = Counter(self.inventory)
            word_counts = Counter(closest_organelle['word'])

            if inventory_counts == word_counts:
                # Success
                closest_organelle['activated'] = True
                self.inventory.clear()
                self._create_particles(closest_organelle['pos'], 50, self.ORGANELLE_COLORS[closest_organelle['name']][0], 3)
                # sfx: organelle_activate_success

                if closest_organelle['activates_id'] > 0:
                    for plat in self.platforms:
                        if plat.get('id') == closest_organelle['activates_id']:
                            plat['active'] = True
                elif closest_organelle['activates_id'] == -1: # Win condition
                    self.win = True

                self.score += 100
                return 1.0
            else:
                # Failure
                self._create_particles(closest_organelle['pos'], 20, self.COLOR_FAIL, 1.5)
                # sfx: organelle_activate_fail
                self.score -= 50
                return -0.5
        return 0.0

    def _create_particles(self, pos, count, color, speed_mult=2, angle_range=None):
        for _ in range(count):
            if angle_range:
                angle = random.uniform(angle_range[0], angle_range[1])
            else:
                angle = random.uniform(0, 2 * math.pi)
            
            speed = random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': random.uniform(1, 4),
                'color': color,
                'life': self.PARTICLE_LIFESPAN
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG_DARK)
        for y in range(self.SCREEN_HEIGHT // 20):
            color_lerp = y / (self.SCREEN_HEIGHT // 20)
            rect_color = (
                self.COLOR_BG_DARK[0] * (1 - color_lerp) + self.COLOR_BG_LIGHT[0] * color_lerp,
                self.COLOR_BG_DARK[1] * (1 - color_lerp) + self.COLOR_BG_LIGHT[1] * color_lerp,
                self.COLOR_BG_DARK[2] * (1 - color_lerp) + self.COLOR_BG_LIGHT[2] * color_lerp,
            )
            pygame.draw.rect(self.screen, rect_color, (0, y * 20, self.SCREEN_WIDTH, 20))
        
        for p in self.bg_particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*self.COLOR_BG_LIGHT, 100))

        # --- Game Elements ---
        self._render_game()
        
        # --- UI Overlay ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Platforms
        for plat in self.platforms:
            if plat['active']:
                color = self.COLOR_PLATFORM_ACTIVE if plat['type'] == 'dynamic' else self.COLOR_PLATFORM
                pygame.draw.rect(self.screen, color, plat['rect'], border_radius=3)
                pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), plat['rect'], width=2, border_radius=3)

        # Organelles
        for org in self.organelles:
            color_active, color_inactive = self.ORGANELLE_COLORS[org['name']]
            color = color_active if org['activated'] else color_inactive
            radius = 20
            pulse = (math.sin(self.steps * 0.1 + org['pos'][0]) * 0.5 + 0.5) * 5 if not org['activated'] else 0
            
            pygame.gfxdraw.filled_circle(self.screen, int(org['pos'][0]), int(org['pos'][1]), int(radius + pulse), (*color, 50))
            pygame.gfxdraw.aacircle(self.screen, int(org['pos'][0]), int(org['pos'][1]), int(radius + pulse), color)
            pygame.gfxdraw.filled_circle(self.screen, int(org['pos'][0]), int(org['pos'][1]), int(radius * 0.7), color)

        # Letters
        for letter in self.letters:
            if not letter['collected']:
                bob = math.sin(letter['bob_offset']) * 3
                pos = (int(letter['pos'][0]), int(letter['pos'][1] + bob))
                pygame.draw.rect(self.screen, self.COLOR_LETTER, (pos[0]-8, pos[1]-8, 16, 16), border_radius=3)
                text = self.font_small.render(letter['char'], True, self.COLOR_BG_DARK)
                text_rect = text.get_rect(center=pos)
                self.screen.blit(text, text_rect)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / self.PARTICLE_LIFESPAN))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

        # Player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        for i in range(5, 0, -1):
            alpha = 100 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, player_x, player_y - self.PLAYER_SIZE, self.PLAYER_SIZE + i, (*self.COLOR_PLAYER_GLOW, alpha))
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Inventory
        inv_width = len(self.inventory) * 30 + 10
        inv_surf = pygame.Surface((inv_width, 40), pygame.SRCALPHA)
        inv_surf.fill(self.COLOR_UI_BG)
        for i, char in enumerate(self.inventory):
            text = self.font_medium.render(char, True, self.COLOR_LETTER)
            text_rect = text.get_rect(center=(i * 30 + 20, 20))
            inv_surf.blit(text, text_rect)
        self.screen.blit(inv_surf, (self.SCREEN_WIDTH/2 - inv_width/2, 10))

        # Target Word
        closest_organelle, min_dist = None, float('inf')
        for org in self.organelles:
            if not org['activated']:
                dist = np.linalg.norm(self.player_pos - org['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_organelle = org
        
        if closest_organelle and min_dist < self.ACTIVATION_RANGE * 2:
            org_color = self.ORGANELLE_COLORS[closest_organelle['name']][0]
            word_text = self.font_medium.render(f"TARGET: {closest_organelle['word']}", True, org_color)
            text_rect = word_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT - 30))
            self.screen.blit(word_text, text_rect)

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "MISSION COMPLETE" if self.win else "GAME OVER"
            color = self.COLOR_SUCCESS if self.win else self.COLOR_FAIL
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "inventory_size": len(self.inventory),
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the evaluation server.
    
    # Un-dummy the video driver for local rendering
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    pygame.display.set_caption("Cell Explorer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    while not (terminated or truncated):
        movement = 0 # None
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(2000) # Pause to show final screen
            obs, info = env.reset()
            terminated = False
            truncated = False

        clock.tick(30) # Run at 30 FPS
        
    env.close()