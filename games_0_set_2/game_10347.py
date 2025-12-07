import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:19:59.763629
# Source Brief: brief_00347.md
# Brief Index: 347
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

def lerp(a, b, t):
    """Linear interpolation"""
    return a + (b - a) * t

def draw_glowing_circle(surface, color, center, radius, glow_strength):
    """Draws a circle with a glowing effect."""
    glow_color = tuple(min(255, c + glow_strength) for c in color)
    max_glow_radius = int(radius * 1.8)
    
    for i in range(max_glow_radius, int(radius), -1):
        alpha = int(255 * (1 - (i - radius) / (max_glow_radius - radius))**2 * 0.3)
        if alpha > 0:
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), i, (*glow_color, alpha))
    
    pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)
    pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)

def draw_lowpoly_rect(surface, color, rect, depth=5):
    """Draws a rectangle with a faux 3D/low-poly look."""
    x, y, w, h = rect
    main_face = (x + depth, y, w - depth, h - depth)
    top_face = [(x, y + h), (x + depth, y + h - depth), (x + w, y + h - depth), (x + w - depth, y + h)]
    left_face = [(x, y), (x + depth, y), (x + depth, y + h - depth), (x, y + h)]
    
    darker = tuple(max(0, c - 40) for c in color)
    darkest = tuple(max(0, c - 60) for c in color)

    pygame.draw.rect(surface, color, main_face)
    pygame.draw.polygon(surface, darker, top_face)
    pygame.draw.polygon(surface, darkest, left_face)


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-3, -1)
        self.life = random.randint(20, 40)
        self.radius = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity
        self.life -= 1
        self.radius -= 0.05

    def draw(self, surface):
        if self.life > 0 and self.radius > 0:
            alpha = int(255 * (self.life / 40))
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), (*self.color, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Restore an ancient temple by magnetically guiding stone heads to their altars. "
        "Collect lost code fragments and switch between sizes to navigate the ruins."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to grow larger and magnetically attract stone heads. "
        "Press shift to shrink and fit through tight spaces."
    )
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
    MAX_STEPS = 1000
    INTERPOLATION_SPEED = 0.25

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_PLAYER = (220, 50, 50)
    COLOR_WALL = (60, 70, 80)
    COLOR_GOAL_INCOMPLETE = (0, 100, 80)
    COLOR_GOAL_COMPLETE = (80, 220, 200)
    COLOR_HEAD = (150, 150, 160)
    COLOR_FRAGMENT = (255, 220, 50)
    COLOR_MAGNETIC_FIELD = (50, 150, 255)
    COLOR_UI_TEXT = (230, 230, 230)

    # Player params
    SMALL_PLAYER_RADIUS = 8
    LARGE_PLAYER_RADIUS = 18
    MAGNET_RANGE = 4.5  # in grid units

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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_code = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_win = pygame.font.SysFont("Verdana", 48, bold=True)
        
        # State variables initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message_alpha = 0

        self.player_pos = [0, 0]
        self.player_visual_pos = [0, 0]
        self.player_size_state = 'small'
        self.player_visual_radius = self.SMALL_PLAYER_RADIUS
        
        self.stone_heads = []
        self.temple_sections = []
        self.code_fragments = []
        self.walls = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.prev_head_distances = {}
        self.prev_fragment_distance = 0.0
        
        # self.reset() # reset is called by the wrapper or user
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message_alpha = 0
        self.particles.clear()
        
        # --- Level Design ---
        self.player_pos = [2, 5]
        self.player_visual_pos = [p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in self.player_pos]
        self.player_size_state = 'small'
        self.player_visual_radius = self.SMALL_PLAYER_RADIUS
        
        wall_map = [
            "################",
            "#              #",
            "# H ##### F    #",
            "# G #   #      #",
            "#   #   #      #",
            "# F ##### H    #",
            "#   #   # G    #",
            "#   #   #      #",
            "#       F      #",
            "################",
        ]
        self.walls = []
        self.stone_heads = []
        self.temple_sections = []
        self.code_fragments = []
        head_id_counter = 0
        for r, row_str in enumerate(wall_map):
            for c, char in enumerate(row_str):
                pos = [c, r]
                if char == '#': self.walls.append(pos)
                elif char == 'H':
                    self.stone_heads.append({
                        "id": head_id_counter, "pos": pos, 
                        "visual_pos": [p * self.CELL_SIZE + self.CELL_SIZE/2 for p in pos]
                    })
                    head_id_counter += 1
                elif char == 'G':
                    self.temple_sections.append({"id": len(self.temple_sections), "pos": pos, "restored": False})
                elif char == 'F':
                    self.code_fragments.append({"pos": pos, "collected": False, "glow_phase": random.uniform(0, 2 * math.pi)})

        self.prev_space_held = False
        self.prev_shift_held = False

        self._update_distance_references()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Handling ---
        grow_action = space_held and not self.prev_space_held
        shrink_action = shift_held and not self.prev_shift_held

        if grow_action and self.player_size_state == 'small':
            self.player_size_state = 'large'
            # sound: player_grow.wav
        elif shrink_action and self.player_size_state == 'large':
            self.player_size_state = 'small'
            # sound: player_shrink.wav
        
        if movement != 0:
            # sound: player_teleport.wav
            original_pos = list(self.player_pos)
            if movement == 1: self.player_pos[1] -= 1  # Up
            elif movement == 2: self.player_pos[1] += 1 # Down
            elif movement == 3: self.player_pos[0] -= 1 # Left
            elif movement == 4: self.player_pos[0] += 1 # Right
            
            # Collision detection
            is_wall = self.player_pos in self.walls
            is_small_gap = self.player_pos in [h['pos'] for h in self.stone_heads] and self.player_size_state == 'large'
            if is_wall or is_small_gap:
                self.player_pos = original_pos # Revert move

        # --- Game Logic ---
        # 1. Magnetism
        if self.player_size_state == 'large':
            for head in self.stone_heads:
                dist = math.hypot(self.player_pos[0] - head['pos'][0], self.player_pos[1] - head['pos'][1])
                if dist < self.MAGNET_RANGE and dist > 0:
                    # sound: magnetism_pull.wav
                    dx = self.player_pos[0] - head['pos'][0]
                    dy = self.player_pos[1] - head['pos'][1]
                    
                    new_head_pos = list(head['pos'])
                    if abs(dx) > abs(dy):
                        new_head_pos[0] += int(np.sign(dx))
                    else:
                        new_head_pos[1] += int(np.sign(dy))
                    
                    if new_head_pos not in self.walls and new_head_pos not in [h['pos'] for h in self.stone_heads if h != head]:
                        head['pos'] = new_head_pos

        # 2. Fragment Collection
        for frag in self.code_fragments:
            if not frag['collected'] and self.player_pos == frag['pos']:
                frag['collected'] = True
                self.score += 5
                reward += 5
                # sound: collect_fragment.wav

        # 3. Temple Restoration
        for section in self.temple_sections:
            if not section['restored']:
                for head in self.stone_heads:
                    if head['id'] == section['id'] and head['pos'] == section['pos']:
                        section['restored'] = True
                        self.score += 10
                        reward += 10
                        # sound: restore_section.wav
                        px, py = (p * self.CELL_SIZE + self.CELL_SIZE/2 for p in section['pos'])
                        for _ in range(30):
                            self.particles.append(Particle(px, py, self.COLOR_GOAL_COMPLETE))

        # --- Reward Calculation ---
        reward += self._calculate_continuous_reward()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and self._is_win():
            reward += 100

        # Update previous action state for next step
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self._update_distance_references()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_distance_references(self):
        # For heads
        for head in self.stone_heads:
            section = next(s for s in self.temple_sections if s['id'] == head['id'])
            if not section['restored']:
                dist = math.hypot(head['pos'][0] - section['pos'][0], head['pos'][1] - section['pos'][1])
                self.prev_head_distances[head['id']] = dist
        
        # For fragments
        uncollected = [f for f in self.code_fragments if not f['collected']]
        if uncollected:
            closest_frag = min(uncollected, key=lambda f: math.hypot(self.player_pos[0] - f['pos'][0], self.player_pos[1] - f['pos'][1]))
            self.prev_fragment_distance = math.hypot(self.player_pos[0] - closest_frag['pos'][0], self.player_pos[1] - closest_frag['pos'][1])
        else:
            self.prev_fragment_distance = 0

    def _calculate_continuous_reward(self):
        reward = 0.0
        # Head movement reward
        for head in self.stone_heads:
            section = next(s for s in self.temple_sections if s['id'] == head['id'])
            if not section['restored']:
                new_dist = math.hypot(head['pos'][0] - section['pos'][0], head['pos'][1] - section['pos'][1])
                prev_dist = self.prev_head_distances.get(head['id'], new_dist)
                if new_dist < prev_dist:
                    reward += 0.1 * (prev_dist - new_dist)

        # Fragment proximity reward
        uncollected = [f for f in self.code_fragments if not f['collected']]
        if uncollected:
            closest_frag = min(uncollected, key=lambda f: math.hypot(self.player_pos[0] - f['pos'][0], self.player_pos[1] - f['pos'][1]))
            new_dist = math.hypot(self.player_pos[0] - closest_frag['pos'][0], self.player_pos[1] - closest_frag['pos'][1])
            if new_dist < self.prev_fragment_distance:
                reward += 1.0 * (self.prev_fragment_distance - new_dist)
        
        return reward

    def _is_win(self):
        all_restored = all(s['restored'] for s in self.temple_sections)
        all_collected = all(f['collected'] for f in self.code_fragments)
        return all_restored and all_collected

    def _check_termination(self):
        if self._is_win():
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Interpolate visual positions ---
        target_radius = self.LARGE_PLAYER_RADIUS if self.player_size_state == 'large' else self.SMALL_PLAYER_RADIUS
        self.player_visual_radius = lerp(self.player_visual_radius, target_radius, self.INTERPOLATION_SPEED)
        
        target_player_visual_pos = [p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in self.player_pos]
        self.player_visual_pos[0] = lerp(self.player_visual_pos[0], target_player_visual_pos[0], self.INTERPOLATION_SPEED)
        self.player_visual_pos[1] = lerp(self.player_visual_pos[1], target_player_visual_pos[1], self.INTERPOLATION_SPEED)
        
        for head in self.stone_heads:
            target_head_visual_pos = [p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in head['pos']]
            head['visual_pos'][0] = lerp(head['visual_pos'][0], target_head_visual_pos[0], self.INTERPOLATION_SPEED)
            head['visual_pos'][1] = lerp(head['visual_pos'][1], target_head_visual_pos[1], self.INTERPOLATION_SPEED)

        # --- Draw elements ---
        for wall_pos in self.walls:
            rect = (wall_pos[0] * self.CELL_SIZE, wall_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            draw_lowpoly_rect(self.screen, self.COLOR_WALL, rect)
            
        for section in self.temple_sections:
            color = self.COLOR_GOAL_COMPLETE if section['restored'] else self.COLOR_GOAL_INCOMPLETE
            center_x = section['pos'][0] * self.CELL_SIZE + self.CELL_SIZE / 2
            center_y = section['pos'][1] * self.CELL_SIZE + self.CELL_SIZE / 2
            pygame.gfxdraw.box(self.screen, (int(center_x - 12), int(center_y - 12), 24, 24), (*color, 80))
            pygame.gfxdraw.rectangle(self.screen, (int(center_x - 12), int(center_y - 12), 24, 24), color)

        for frag in self.code_fragments:
            if not frag['collected']:
                frag['glow_phase'] += 0.1
                glow = 50 + 20 * math.sin(frag['glow_phase'])
                center = [p * self.CELL_SIZE + self.CELL_SIZE/2 for p in frag['pos']]
                draw_glowing_circle(self.screen, self.COLOR_FRAGMENT, center, 6, glow)

        if self.player_size_state == 'large':
            for head in self.stone_heads:
                dist = math.hypot(self.player_pos[0] - head['pos'][0], self.player_pos[1] - head['pos'][1])
                if dist < self.MAGNET_RANGE:
                    alpha = int(150 * (1 - dist / self.MAGNET_RANGE))
                    pygame.draw.aaline(self.screen, (*self.COLOR_MAGNETIC_FIELD, alpha), self.player_visual_pos, head['visual_pos'], 2)

        for head in self.stone_heads:
            rect = (head['visual_pos'][0] - 12, head['visual_pos'][1] - 12, 24, 24)
            draw_lowpoly_rect(self.screen, self.COLOR_HEAD, rect, depth=3)

        draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.player_visual_pos, self.player_visual_radius, 40)
        
        # Particles
        for p in self.particles[:]:
            p.update()
            p.draw(self.screen)
            if p.life <= 0:
                self.particles.remove(p)

    def _render_ui(self):
        # Score / Restored sections
        restored_count = sum(1 for s in self.temple_sections if s['restored'])
        total_sections = len(self.temple_sections)
        score_text = self.font_ui.render(f"Restored: {restored_count}/{total_sections}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Code fragments
        code_symbols = "ᛗᚿᛟ" # Mayan-esque symbols
        code_display = ""
        for i, frag in enumerate(self.code_fragments):
            code_display += code_symbols[i % len(code_symbols)] if frag['collected'] else "_"
            code_display += " "
        
        code_text = self.font_code.render(code_display, True, self.COLOR_FRAGMENT)
        text_rect = code_text.get_rect(bottomright=(self.SCREEN_WIDTH - 15, self.SCREEN_HEIGHT - 10))
        self.screen.blit(code_text, text_rect)

        # Win Message
        if self._is_win():
            self.win_message_alpha = min(255, self.win_message_alpha + 15)
            win_text = self.font_win.render("TEMPLE RESTORED", True, self.COLOR_GOAL_COMPLETE)
            win_text.set_alpha(self.win_message_alpha)
            text_rect = win_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(win_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "restored_sections": sum(1 for s in self.temple_sections if s['restored']),
            "collected_fragments": sum(1 for f in self.code_fragments if f['collected']),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block is for human play and debugging.
    # It will not be executed by the evaluation system.
    # Set the SDL_VIDEODRIVER to a real driver to see the window.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    
    pygame.display.set_caption("Mayan Temple Explorer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause to see final state
            obs, info = env.reset()

        env.clock.tick(30) # Limit to 30 FPS for smooth human play

    env.close()