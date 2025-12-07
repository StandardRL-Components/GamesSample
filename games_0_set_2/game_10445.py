import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:24:38.336215
# Source Brief: brief_00445.md
# Brief Index: 445
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A stealth-action puzzle platformer where a robotic cat burglar uses
    gravity-altering cards to navigate intricate clockwork mansions,
    silently taking down security bots for bonus points.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    game_description = (
        "Navigate a clockwork mansion as a robotic cat burglar, using gravity-altering cards to "
        "evade security bots and reach the exit."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to take down an adjacent bot. "
        "Hold shift to use a gravity-altering card."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 40
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 1000
        self.render_mode = render_mode

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 50)

        # --- Colors (Steampunk Aesthetic) ---
        self.COLOR_BG = (20, 15, 25)
        self.COLOR_GEAR_BG = (30, 25, 35)
        self.COLOR_WALL = (70, 60, 80)
        self.COLOR_PLAYER = (224, 122, 53) # Copper
        self.COLOR_PLAYER_GLOW = (224, 122, 53, 50)
        self.COLOR_BOT = (150, 160, 170) # Steel Gray
        self.COLOR_BOT_EYE = (255, 0, 0)
        self.COLOR_VISION_CONE = (255, 0, 0, 40)
        self.COLOR_TARGET = (255, 215, 0) # Gold
        self.COLOR_GRAVITY_EFFECT = (50, 150, 255) # Blue
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.CARD_COLORS = {
            (0, -1): (100, 180, 255),  # Up: Light Blue
            (0, 1): (100, 255, 180),   # Down: Light Green
            (-1, 0): (255, 180, 100),  # Left: Light Orange
            (1, 0): (255, 100, 180),   # Right: Light Pink
        }

        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.player_facing_dir = None
        self.bots = []
        self.walls = set()
        self.target_pos = None
        self.gravity = None
        self.cards = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_takedown_step = -10
        self.particles = []
        self.gravity_effect_timer = 0
        self.background_gears = []

        # Use a numpy random number generator
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = (0, 1) # Default gravity is down
        self.last_takedown_step = -10
        self.particles = []
        self.gravity_effect_timer = 0

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Level layout
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        # Internal platforms
        for x in range(4, 12): self.walls.add((x, 4))
        for y in range(5, 8): self.walls.add((8, y))

        self.player_pos = (2, 2)
        self.player_facing_dir = (1, 0)
        self.target_pos = (14, 8)

        self.cards = [(-1, 0), (0, -1), (1, 0)] # Left, Up, Right cards

        self.bots = [
            {'pos': (6, 3), 'path': [(6, 3), (7, 3), (8, 3), (9, 3), (10, 3)], 'path_idx': 0, 'dir': 1, 'facing': (1,0)},
            {'pos': (12, 6), 'path': [(12, 6), (12, 7), (12, 8), (11, 8), (10, 8)], 'path_idx': 0, 'dir': 1, 'facing': (0,1)}
        ]

        if not self.background_gears:
            for _ in range(20):
                self.background_gears.append({
                    'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                    'radius': self.np_random.integers(15, 61),
                    'speed': self.np_random.uniform(-0.5, 0.5),
                    'angle': self.np_random.uniform(0, 360)
                })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # 1. Handle player actions
        if shift_held and len(self.cards) > 0 and self.gravity_effect_timer == 0:
            self.gravity = self.cards.pop(0)
            self.gravity_effect_timer = 15 # Visual effect lasts 15 frames/steps
            # SFX: Gravity Shift Whoosh

        if space_held:
            takedown_reward = self._handle_takedown()
            if takedown_reward > 0:
                reward += takedown_reward
                # SFX: Robot Takedown Zap

        self._handle_player_movement(movement)

        # 2. Update game world
        self._update_bots()
        self._update_particles()
        if self.gravity_effect_timer > 0:
            self.gravity_effect_timer -= 1

        # 3. Check for game-ending conditions
        terminated = False
        if self._check_detection():
            reward = -100.0
            terminated = True
            # SFX: Alert!
        elif self.player_pos == self.target_pos:
            reward += 50.0
            self.score += 50
            terminated = True
            # SFX: Success!
        elif self.steps >= self.MAX_STEPS - 1:
            terminated = True

        if not terminated:
            reward += 0.1 # Survival reward

        self.game_over = terminated
        self.steps += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_takedown(self):
        takedown_pos = (self.player_pos[0] + self.player_facing_dir[0],
                        self.player_pos[1] + self.player_facing_dir[1])

        bot_to_remove = None
        for bot in self.bots:
            if bot['pos'] == takedown_pos:
                bot_to_remove = bot
                break

        if bot_to_remove:
            self.bots.remove(bot_to_remove)
            reward = 5.0
            self.score += 5
            if self.steps - self.last_takedown_step <= 3: # Combo bonus
                reward += 2.0
                self.score += 2
            self.last_takedown_step = self.steps
            self._create_takedown_particles(takedown_pos)
            return reward
        return 0

    def _handle_player_movement(self, movement_action):
        # First, apply gravity
        grav_pos = (self.player_pos[0] + self.gravity[0], self.player_pos[1] + self.gravity[1])
        if grav_pos not in self.walls:
            self.player_pos = grav_pos

        # Then, apply player-intended movement
        move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement_action)
        if move_dir:
            self.player_facing_dir = move_dir
            next_pos = (self.player_pos[0] + move_dir[0], self.player_pos[1] + move_dir[1])
            if next_pos not in self.walls:
                self.player_pos = next_pos

    def _update_bots(self):
        for bot in self.bots:
            path = bot['path']
            if not path: continue

            bot['path_idx'] += bot['dir']
            if not (0 <= bot['path_idx'] < len(path)):
                bot['dir'] *= -1
                bot['path_idx'] += bot['dir'] * 2

            new_pos = path[bot['path_idx']]
            old_pos = bot['pos']
            bot['pos'] = new_pos

            if new_pos[0] > old_pos[0]: bot['facing'] = (1, 0)
            elif new_pos[0] < old_pos[0]: bot['facing'] = (-1, 0)
            elif new_pos[1] > old_pos[1]: bot['facing'] = (0, 1)
            elif new_pos[1] < old_pos[1]: bot['facing'] = (0, -1)

    def _check_detection(self):
        for bot in self.bots:
            for i in range(1, 4): # Vision cone is 3 tiles long
                vx = bot['pos'][0] + bot['facing'][0] * i
                vy = bot['pos'][1] + bot['facing'][1] * i
                if (vx, vy) == self.player_pos:
                    return True
                if (vx, vy) in self.walls:
                    break
        return False

    def _create_takedown_particles(self, pos_grid):
        px, py = (pos_grid[0] + 0.5) * self.TILE_SIZE, (pos_grid[1] + 0.5) * self.TILE_SIZE
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 31),
                'color': random.choice([(255,255,255), (255,215,0), self.COLOR_BOT])
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_walls()
        self._render_target()
        self._render_bots_and_vision()
        self._render_player()
        self._render_effects()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for gear in self.background_gears:
            gear['angle'] += gear['speed']
            self._draw_gear(gear['pos'], gear['radius'], 10, gear['angle'], self.COLOR_GEAR_BG)

    def _draw_gear(self, pos, radius, num_teeth, angle_deg, color):
        angle_rad = math.radians(angle_deg)
        points = []
        tooth_depth = radius * 0.15
        for i in range(num_teeth * 2):
            r = radius if i % 2 == 0 else radius - tooth_depth
            a = angle_rad + i * math.pi / num_teeth
            points.append((int(pos[0] + r * math.cos(a)), int(pos[1] + r * math.sin(a))))
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius * 0.5), self.COLOR_BG)

    def _render_walls(self):
        for x, y in self.walls:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

    def _render_target(self):
        x, y = self.target_pos
        rect = pygame.Rect(x * self.TILE_SIZE + 5, y * self.TILE_SIZE + 5, self.TILE_SIZE - 10, self.TILE_SIZE - 10)
        pygame.draw.rect(self.screen, self.COLOR_TARGET, rect, border_radius=4)
        pygame.draw.circle(self.screen, self.COLOR_BG, rect.center, 8)
        pygame.draw.circle(self.screen, self.COLOR_TARGET, rect.center, 8, 2)

    def _render_bots_and_vision(self):
        for bot in self.bots:
            # Vision cone
            p1 = ((bot['pos'][0] + 0.5) * self.TILE_SIZE, (bot['pos'][1] + 0.5) * self.TILE_SIZE)
            end_tile = bot['pos']
            for i in range(1, 4):
                next_tile = (bot['pos'][0] + bot['facing'][0] * i, bot['pos'][1] + bot['facing'][1] * i)
                if next_tile in self.walls: break
                end_tile = next_tile

            p2_offset = (-bot['facing'][1], -bot['facing'][0])
            p2 = ((end_tile[0] + 0.5 + p2_offset[0]) * self.TILE_SIZE, (end_tile[1] + 0.5 + p2_offset[1]) * self.TILE_SIZE)
            p3 = ((end_tile[0] + 0.5 - p2_offset[0]) * self.TILE_SIZE, (end_tile[1] + 0.5 - p2_offset[1]) * self.TILE_SIZE)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_VISION_CONE)

            # Bot body
            x, y = bot['pos']
            rect = pygame.Rect(x * self.TILE_SIZE + 8, y * self.TILE_SIZE + 8, self.TILE_SIZE - 16, self.TILE_SIZE - 16)
            pygame.draw.rect(self.screen, self.COLOR_BOT, rect, border_radius=3)
            # Bot eye
            eye_pos = (rect.centerx + bot['facing'][0] * 5, rect.centery + bot['facing'][1] * 5)
            pygame.draw.circle(self.screen, self.COLOR_BOT_EYE, eye_pos, 3)

    def _render_player(self):
        x, y = self.player_pos
        center_x = int((x + 0.5) * self.TILE_SIZE)
        center_y = int((y + 0.5) * self.TILE_SIZE)

        # Glow
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 18, self.COLOR_PLAYER_GLOW)

        # Body
        body_rect = pygame.Rect(0, 0, 24, 24)
        body_rect.center = (center_x, center_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=5)

        # Ears
        ear1 = ((center_x - 8, center_y - 12), (center_x - 12, center_y - 16), (center_x - 4, center_y - 16))
        ear2 = ((center_x + 8, center_y - 12), (center_x + 4, center_y - 16), (center_x + 12, center_y - 16))
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ear1)
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ear2)

    def _render_effects(self):
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, int(p['life'] / 5)))

        # Gravity effect
        if self.gravity_effect_timer > 0:
            alpha = int(100 * (self.gravity_effect_timer / 15))
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_GRAVITY_EFFECT, alpha), (0, 0, self.WIDTH, self.HEIGHT), width=10)
            self.screen.blit(s, (0,0))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Cards
        card_text = self.font_ui.render("CARDS:", True, self.COLOR_UI_TEXT)
        self.screen.blit(card_text, (self.WIDTH - 180, 15))
        for i, card_dir in enumerate(self.cards):
            rect = pygame.Rect(self.WIDTH - 90 + i * 25, 15, 20, 20)
            pygame.draw.rect(self.screen, self.CARD_COLORS[card_dir], rect, border_radius=3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "gravity": self.gravity,
            "cards_left": len(self.cards),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For the main script, we need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Clockwork Cat Burglar")

    terminated = False
    running = True
    clock = pygame.time.Clock()

    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                 obs, info = env.reset()
                 terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)

            # print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        if terminated:
            s = pygame.Surface((env.WIDTH, env.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            screen.blit(s, (0,0))

            end_text = env.font_title.render("GAME OVER", True, (255, 50, 50))
            sub_text = env.font_ui.render("Press 'R' to Restart", True, (255, 255, 255))

            text_rect = end_text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 - 20))
            sub_rect = sub_text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 + 20))

            screen.blit(end_text, text_rect)
            screen.blit(sub_text, sub_rect)

        pygame.display.flip()
        clock.tick(env.metadata["render_fps"])

    env.close()