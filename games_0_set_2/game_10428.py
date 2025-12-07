import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    GameEnv: A puzzle/stealth game where the agent must navigate a grid.
    The agent places 'quark cards' to complete patterns, which activates portals.
    The goal is to move the avatar through these portals to reach the nucleus,
    while avoiding mobile gluon detectors.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a grid to reach the nucleus. Place quark cards to complete patterns and activate portals, "
        "but beware of the patrolling gluon detectors."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press 'C' to cycle through your hand. "
        "Press space to place a quark card and shift to activate a portal."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 8
    GRID_TOP_MARGIN, GRID_LEFT_MARGIN = 40, 40
    CELL_SIZE = 40
    
    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 50, 80)
    COLOR_AVATAR = (0, 255, 255)
    COLOR_NUCLEUS = (255, 255, 100)
    COLOR_DETECTOR = (255, 50, 50)
    COLOR_DETECTOR_GLOW = (255, 100, 100)
    COLOR_PORTAL_INACTIVE = (200, 100, 0)
    COLOR_PORTAL_ACTIVE = (255, 180, 0)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    QUARK_DATA = {
        1: {"name": "Up", "color": (50, 150, 255)},
        2: {"name": "Down", "color": (50, 255, 150)},
        3: {"name": "Strange", "color": (255, 255, 50)},
        4: {"name": "Charm", "color": (200, 100, 255)},
    }
    
    # Game Parameters
    MAX_STEPS = 1000
    DETECTOR_SPEED_INCREASE_INTERVAL = 50
    QUARK_UNLOCK_THRESHOLD = 2 # Portals activated to unlock next quark type
    INITIAL_HAND_SIZE = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48)
        
        # Initialize state variables to None, to be set in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.victory = None
        self.grid = None
        self.avatar_pos = None
        self.nucleus_pos = None
        self.cursor_pos = None
        self.portals = None
        self.detectors = None
        self.hand = None
        self.deck = None
        self.unlocked_quarks = None
        self.portals_activated_count = None
        self.particles = None
        self.selected_card_idx = None
        self.last_action_feedback = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        self.avatar_pos = (0, self.GRID_HEIGHT // 2)
        self.nucleus_pos = (self.GRID_WIDTH - 1, self.GRID_HEIGHT // 2)

        self.portals = self._initialize_portals()
        self.detectors = self._initialize_detectors()
        
        self.unlocked_quarks = [1, 2]
        self.deck = self._create_deck()
        self.hand = [self.deck.pop() for _ in range(self.INITIAL_HAND_SIZE) if self.deck]
        self.selected_card_idx = 0

        self.portals_activated_count = 0
        self.particles = []
        self.last_action_feedback = {"text": "", "timer": 0, "color": (255,255,255)}

        return self._get_observation(), self._get_info()

    def _initialize_portals(self):
        return [
            {"pos": (3, self.GRID_HEIGHT // 2), "active": False, "pattern": [((0, -1), 1), ((0, 1), 2)]}, # Up, Down
            {"pos": (6, 2), "active": False, "pattern": [((-1, 0), 3), ((1, 0), 3)]}, # Strange, Strange
            {"pos": (6, self.GRID_HEIGHT - 3), "active": False, "pattern": [((-1, 0), 4), ((1, 0), 4)]}, # Charm, Charm
        ]

    def _initialize_detectors(self):
        return [{
            "pos": (self.GRID_WIDTH // 2, 0),
            "path": [(self.GRID_WIDTH // 2, y) for y in range(self.GRID_HEIGHT)] + \
                    [(self.GRID_WIDTH // 2, y) for y in range(self.GRID_HEIGHT - 2, 0, -1)],
            "path_index": 0,
            "move_counter": 0,
            "move_threshold": 30 # Slower start
        }]

    def _create_deck(self):
        deck = self.unlocked_quarks * 15
        self.np_random.shuffle(deck)
        return deck

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        self._handle_movement(movement)
        if space_press: reward += self._handle_place_card()
        if shift_press: reward += self._handle_activate_portal()
        
        # --- Update Game State ---
        self._update_detectors()
        self._update_particles()
        if self.last_action_feedback["timer"] > 0:
            self.last_action_feedback["timer"] -= 1

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.avatar_pos == self.nucleus_pos:
            reward = 100.0
            terminated = True
            self.game_over = True
            self.victory = True
        
        for detector in self.detectors:
            if detector["pos"] == self.avatar_pos:
                reward = -10.0
                terminated = True
                self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        # Action 0 is 'cycle selected card' instead of 'no-op' for better gameplay
        if movement == 0:
            if self.hand:
                self.selected_card_idx = (self.selected_card_idx + 1) % len(self.hand)
        elif movement == 1: # Up
            self.cursor_pos = (self.cursor_pos[0], max(0, self.cursor_pos[1] - 1))
        elif movement == 2: # Down
            self.cursor_pos = (self.cursor_pos[0], min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1))
        elif movement == 3: # Left
            self.cursor_pos = (max(0, self.cursor_pos[0] - 1), self.cursor_pos[1])
        elif movement == 4: # Right
            self.cursor_pos = (min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1), self.cursor_pos[1])

    def _handle_place_card(self):
        if not self.hand:
            self._set_feedback("No cards left!", (255, 100, 100))
            return -0.1

        x, y = self.cursor_pos
        is_portal_pos = any(p["pos"] == self.cursor_pos for p in self.portals)
        is_nucleus_pos = self.cursor_pos == self.nucleus_pos
        is_avatar_pos = self.cursor_pos == self.avatar_pos

        if self.grid[x, y] == 0 and not is_portal_pos and not is_nucleus_pos and not is_avatar_pos:
            card_to_place = self.hand.pop(self.selected_card_idx)
            self.grid[x, y] = card_to_place
            if self.deck:
                self.hand.append(self.deck.pop())
            if self.hand:
                self.selected_card_idx = min(self.selected_card_idx, len(self.hand) - 1)
            else:
                self.selected_card_idx = 0
            
            self._create_particles(self.cursor_pos, self.QUARK_DATA[card_to_place]['color'], 10)
            self._set_feedback(f"Placed {self.QUARK_DATA[card_to_place]['name']} Quark", (200, 255, 200))
            return 0.1
        else:
            self._set_feedback("Invalid placement!", (255, 100, 100))
            return -0.1

    def _handle_activate_portal(self):
        portal = next((p for p in self.portals if p["pos"] == self.cursor_pos and not p["active"]), None)
        if not portal:
            self._set_feedback("Not a valid portal", (255, 200, 100))
            return 0.0

        pattern_met = True
        for (dx, dy), quark_id in portal["pattern"]:
            check_x, check_y = portal["pos"][0] + dx, portal["pos"][1] + dy
            if not (0 <= check_x < self.GRID_WIDTH and 0 <= check_y < self.GRID_HEIGHT and self.grid[check_x, check_y] == quark_id):
                pattern_met = False
                break
        
        if pattern_met:
            portal["active"] = True
            self.avatar_pos = portal["pos"]
            self.portals_activated_count += 1
            self._create_particles(self.cursor_pos, self.COLOR_PORTAL_ACTIVE, 50, life=40, speed=3)
            self._set_feedback("Portal Activated!", (255, 255, 150))

            if self.portals_activated_count % self.QUARK_UNLOCK_THRESHOLD == 0:
                if len(self.unlocked_quarks) < len(self.QUARK_DATA):
                    new_quark = len(self.unlocked_quarks) + 1
                    self.unlocked_quarks.append(new_quark)
                    self.deck.extend([new_quark] * 10)
                    self.np_random.shuffle(self.deck)
                    self._set_feedback(f"New Quark Unlocked: {self.QUARK_DATA[new_quark]['name']}!", (200, 150, 255))

            return 1.0
        else:
            self._set_feedback("Pattern not matched!", (255, 100, 100))
            return -0.2

    def _update_detectors(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.DETECTOR_SPEED_INCREASE_INTERVAL == 0:
            for d in self.detectors:
                d["move_threshold"] = max(5, d["move_threshold"] - 2)

        for d in self.detectors:
            d["move_counter"] += 1
            if d["move_counter"] >= d["move_threshold"]:
                d["move_counter"] = 0
                d["path_index"] = (d["path_index"] + 1) % len(d["path"])
                d["pos"] = d["path"][d["path_index"]]

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
            "portals_activated": self.portals_activated_count,
            "avatar_pos": self.avatar_pos,
            "cursor_pos": self.cursor_pos,
        }
        
    def _grid_to_screen(self, x, y):
        return (self.GRID_LEFT_MARGIN + x * self.CELL_SIZE, self.GRID_TOP_MARGIN + y * self.CELL_SIZE)

    def _render_game(self):
        self._render_grid_and_paths()
        self._render_particles()
        self._render_portals()
        self._render_nucleus()
        self._render_placed_cards()
        self._render_avatar()
        self._render_detectors()
        self._render_cursor()

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.GRID_LEFT_MARGIN, 10))
        steps_text = self.font_medium.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - self.GRID_LEFT_MARGIN, 10))
        
        # Hand display
        hand_y = self.GRID_TOP_MARGIN + self.GRID_HEIGHT * self.CELL_SIZE + 20
        hand_text = self.font_medium.render("HAND:", True, self.COLOR_TEXT)
        self.screen.blit(hand_text, (self.GRID_LEFT_MARGIN, hand_y))
        for i, card_id in enumerate(self.hand):
            card_x = self.GRID_LEFT_MARGIN + 90 + i * (self.CELL_SIZE + 10)
            rect = pygame.Rect(card_x, hand_y, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.QUARK_DATA[card_id]['color'], rect, border_radius=4)
            self._draw_quark_symbol(self.screen, card_id, rect.center, self.CELL_SIZE // 2)
            if i == self.selected_card_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=4)

        # Action feedback
        if self.last_action_feedback["timer"] > 0:
            feedback_surf = self.font_small.render(self.last_action_feedback["text"], True, self.last_action_feedback["color"])
            pos = (self.SCREEN_WIDTH / 2 - feedback_surf.get_width() / 2, self.SCREEN_HEIGHT - 25)
            self.screen.blit(feedback_surf, pos)

        # Game Over/Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_NUCLEUS if self.victory else self.COLOR_DETECTOR
            msg_surf = self.font_large.render(msg, True, color)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH/2 - msg_surf.get_width()/2, self.SCREEN_HEIGHT/2 - msg_surf.get_height()/2))
    
    def _render_grid_and_paths(self):
        # Grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = self._grid_to_screen(x, 0)
            end = self._grid_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, (start[0] - self.CELL_SIZE/2, start[1] - self.CELL_SIZE/2), (end[0] - self.CELL_SIZE/2, end[1] - self.CELL_SIZE/2))
        for y in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_screen(0, y)
            end = self._grid_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, (start[0] - self.CELL_SIZE/2, start[1] - self.CELL_SIZE/2), (end[0] - self.CELL_SIZE/2, end[1] - self.CELL_SIZE/2))
        
        # Detector paths
        for d in self.detectors:
            for i in range(len(d["path"]) - 1):
                p1 = self._grid_to_screen(d["path"][i][0], d["path"][i][1])
                p2 = self._grid_to_screen(d["path"][i+1][0], d["path"][i+1][1])
                self._draw_dashed_line(self.screen, self.COLOR_DETECTOR, p1, p2, 5)

    def _render_placed_cards(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0:
                    card_id = self.grid[x, y]
                    sx, sy = self._grid_to_screen(x, y)
                    rect = pygame.Rect(sx - self.CELL_SIZE/2 + 3, sy - self.CELL_SIZE/2 + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
                    pygame.draw.rect(self.screen, self.QUARK_DATA[card_id]['color'], rect, border_radius=4)
                    self._draw_quark_symbol(self.screen, card_id, rect.center, self.CELL_SIZE // 3)

    def _render_portals(self):
        for portal in self.portals:
            sx, sy = self._grid_to_screen(portal["pos"][0], portal["pos"][1])
            radius = self.CELL_SIZE // 3
            if portal["active"]:
                glow = (math.sin(self.steps * 0.2) + 1) / 2 * 5 + 2
                self._draw_glowing_circle(self.screen, self.COLOR_PORTAL_ACTIVE, (sx, sy), radius, glow)
            else:
                pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, self.COLOR_PORTAL_INACTIVE)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_PORTAL_INACTIVE)

    def _render_nucleus(self):
        sx, sy = self._grid_to_screen(self.nucleus_pos[0], self.nucleus_pos[1])
        base_radius = self.CELL_SIZE // 2 - 5
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        radius = int(base_radius + pulse * 4)
        glow = 5 + pulse * 5
        self._draw_glowing_circle(self.screen, self.COLOR_NUCLEUS, (sx, sy), radius, glow)
    
    def _render_avatar(self):
        sx, sy = self._grid_to_screen(self.avatar_pos[0], self.avatar_pos[1])
        self._draw_glowing_circle(self.screen, self.COLOR_AVATAR, (sx, sy), self.CELL_SIZE // 4, 10)

    def _render_detectors(self):
        for d in self.detectors:
            sx, sy = self._grid_to_screen(d["pos"][0], d["pos"][1])
            size = self.CELL_SIZE - 10
            rect = pygame.Rect(sx - size/2, sy - size/2, size, size)
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            glow_color = tuple(int(c1 * pulse + c2 * (1-pulse)) for c1, c2 in zip(self.COLOR_DETECTOR_GLOW, self.COLOR_DETECTOR))
            pygame.draw.rect(self.screen, glow_color, rect.inflate(6, 6), border_radius=6)
            pygame.draw.rect(self.screen, self.COLOR_DETECTOR, rect, border_radius=4)

    def _render_cursor(self):
        sx, sy = self._grid_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        rect = pygame.Rect(sx - self.CELL_SIZE/2, sy - self.CELL_SIZE/2, self.CELL_SIZE, self.CELL_SIZE)
        alpha = int(100 + (math.sin(self.steps * 0.2) * 50 + 50))
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, self.COLOR_CURSOR + (alpha,), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 3, border_radius=5)
        self.screen.blit(cursor_surface, rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            radius = int(p['life'] / p['max_life'] * p['size'])
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])
    
    # --- Helper & Utility Functions ---

    def _set_feedback(self, text, color):
        self.last_action_feedback = {"text": text, "timer": 30, "color": color}

    def _create_particles(self, pos_grid, color, count, life=20, speed=2, size=5):
        sx, sy = self._grid_to_screen(pos_grid[0], pos_grid[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, speed)
            self.particles.append({
                'pos': [sx, sy],
                'vel': [math.cos(angle) * s, math.sin(angle) * s],
                'life': life,
                'max_life': life,
                'color': color,
                'size': size
            })
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_quark_symbol(self, surface, quark_id, center, size):
        x, y = center
        s = size * 0.8
        points = []
        if quark_id == 1: # Up
            points = [(x, y - s/2), (x - s/2, y + s/2), (x + s/2, y + s/2)]
        elif quark_id == 2: # Down
            points = [(x, y + s/2), (x - s/2, y - s/2), (x + s/2, y - s/2)]
        elif quark_id == 3: # Strange
            points = [(x - s/2, y - s/2), (x + s/2, y - s/2), (x, y + s/2), (x-s/2, y+s/2)]
        elif quark_id == 4: # Charm
            pygame.gfxdraw.aacircle(surface, int(x), int(y), int(s/2), (0,0,0))
            pygame.gfxdraw.aacircle(surface, int(x), int(y), int(s/2-1), (0,0,0))
        if points:
            pygame.gfxdraw.aapolygon(surface, points, (0, 0, 0, 150))
            pygame.gfxdraw.filled_polygon(surface, points, (0, 0, 0, 150))

    def _draw_glowing_circle(self, surface, color, center, radius, max_glow):
        radius = max(1, int(radius))
        for i in range(max(1, int(max_glow))):
            alpha = 255 * (1 - i / max_glow)
            glow_color = color + (int(alpha / 4),)
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius + i, glow_color)
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), radius, color)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius, color)

    def _draw_dashed_line(self, surf, color, start_pos, end_pos, dash_length=5):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dl = dash_length
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0: return
        dx, dy = dx / dist, dy / dist
        
        for i in range(0, int(dist / (dl * 2))):
            _x1, _y1 = x1 + dx * i * dl * 2, y1 + dy * i * dl * 2
            _x2, _y2 = x1 + dx * (i * dl * 2 + dl), y1 + dy * (i * dl * 2 + dl)
            pygame.draw.aaline(surf, color, (_x1, _y1), (_x2, _y2))


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not work unless you have a display and remove the SDL_VIDEODRIVER dummy setting
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quark Sneak")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        # Poll for events and keys once per frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R'
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_c: # Cycle card on press, not hold
                    movement = 0

        keys = pygame.key.get_pressed()
        if movement == 0: # Only check other movements if 'c' wasn't pressed
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(15) # Control manual play speed

    pygame.quit()