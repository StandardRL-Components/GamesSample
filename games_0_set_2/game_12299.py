import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:39:19.339844
# Source Brief: brief_02299.md
# Brief Index: 2299
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a quintet of transforming creatures
    to collect resources across varied terrains. The goal is to collect 50 resources
    by strategically managing the creatures' movement and transformations.

    **Visuals:**
    - Clean, geometric art style.
    - Terrain is represented by large colored zones (green, blue, brown).
    - Creatures are colored polygons that change shape on transformation.
    - The selected creature has a bright white glowing outline.
    - Resources are small, glowing white orbs.

    **Gameplay:**
    - Control one creature at a time, nudging it with arrow keys.
    - Switch between the 5 creatures using the 'space' action.
    - Creatures have different movement types (Flyer, Swimmer, Crawler) which
      affect their speed on different terrains.
    - Every 30 seconds (900 steps), all creatures transform to a new random type,
      requiring the player to adapt their strategy.
    - Creatures exert a small repulsive force on each other, preventing clumping.

    **Objective:**
    - Collect 50 resources to win.
    - The episode also ends if 5000 steps are reached.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - `actions[1]`: Select next creature (1=press)
    - `actions[2]`: Unused (Shift)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a quintet of transforming creatures to collect resources across varied terrains, "
        "adapting your strategy as they change forms."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the selected creature. Press space to switch to the next creature."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_SCORE = 50
    NUM_CREATURES = 5
    NUM_RESOURCES = 75

    # Colors
    COLOR_BG = (15, 18, 26)
    COLOR_GRASS = (38, 77, 50)
    COLOR_WATER = (40, 65, 107)
    COLOR_DESERT = (115, 95, 65)
    COLOR_RESOURCE = (255, 255, 255)
    COLOR_SELECT_GLOW = (255, 255, 255)
    CREATURE_COLORS = [
        (255, 87, 87),    # Red
        (255, 225, 87),   # Yellow
        (168, 87, 255),   # Purple
        (87, 230, 255),   # Cyan
        (255, 168, 87),   # Orange
    ]

    # Gameplay
    TRANSFORM_INTERVAL = 30 * FPS  # 30 seconds
    CREATURE_NUDGE_FORCE = 0.5
    CREATURE_DRAG = 0.92
    CREATURE_REPULSION = 15.0
    CREATURE_REPULSION_RADIUS = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 50)

        # State variables are initialized in reset()
        self.creatures = []
        self.resources = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_creature_idx = 0
        self.transformation_timer = 0
        self.space_pressed_last_frame = False

        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is a helper, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_creature_idx = 0
        self.transformation_timer = self.TRANSFORM_INTERVAL
        self.space_pressed_last_frame = False

        # Define terrain zones
        self.terrain_rects = {
            "grass": pygame.Rect(0, 0, self.WIDTH, self.HEIGHT),
            "water": pygame.Rect(self.WIDTH // 4, self.HEIGHT // 4, self.WIDTH // 2, self.HEIGHT // 2),
            "desert": pygame.Rect(self.WIDTH - 200, self.HEIGHT - 150, 180, 120),
        }

        # Initialize Creatures
        self.creatures = []
        for i in range(self.NUM_CREATURES):
            self.creatures.append(self._create_creature(i))

        # Initialize Resources
        self.resources = []
        for _ in range(self.NUM_RESOURCES):
            self.resources.append(self._create_resource())

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0

        # --- Handle Actions ---
        # 1. Creature Selection (on press, not hold)
        if space_held and not self.space_pressed_last_frame:
            self.selected_creature_idx = (self.selected_creature_idx + 1) % self.NUM_CREATURES
            # sfx: UI_select.wav
        self.space_pressed_last_frame = space_held

        # 2. Movement Nudge
        selected_creature = self.creatures[self.selected_creature_idx]
        
        # Calculate distance to nearest resource before moving for reward calc
        old_dist_to_resource = self._get_dist_to_nearest_resource(selected_creature['pos'])
        
        nudge = pygame.Vector2(0, 0)
        if movement == 1: nudge.y = -1  # Up
        elif movement == 2: nudge.y = 1   # Down
        elif movement == 3: nudge.x = -1  # Left
        elif movement == 4: nudge.x = 1   # Right
        
        if nudge.length() > 0:
            selected_creature['vel'] += nudge.normalize() * self.CREATURE_NUDGE_FORCE

        # --- Update Game State ---
        self._update_creatures()
        collected_this_step = self._check_resource_collection()
        self._update_transformation()
        
        # --- Calculate Reward ---
        # Reward for collecting resources
        if collected_this_step:
            reward += 1.0 * collected_this_step
            # sfx: collect_resource.wav

        # Reward for moving towards the nearest resource
        new_dist_to_resource = self._get_dist_to_nearest_resource(selected_creature['pos'])
        if new_dist_to_resource < old_dist_to_resource:
            reward += 0.1
        else:
            reward -= 0.01

        # --- Check Termination ---
        self.steps += 1
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _create_creature(self, index):
        return {
            "id": index,
            "pos": pygame.Vector2(random.uniform(20, self.WIDTH - 20), random.uniform(20, self.HEIGHT - 20)),
            "vel": pygame.Vector2(0, 0),
            "color": self.CREATURE_COLORS[index],
            "type": random.choice(["flyer", "swimmer", "crawler"]),
            "radius": 10,
            "trail": [],
            "transform_flash": 0
        }

    def _create_resource(self):
        # Avoid placing resources too close to edges
        x = random.uniform(20, self.WIDTH - 20)
        y = random.uniform(20, self.HEIGHT - 20)
        return {
            "pos": pygame.Vector2(x, y),
            "collected": False,
            "radius": 5,
            "glow": random.uniform(0.5, 1.0),
            "glow_dir": 1
        }
    
    def _get_terrain_at_pos(self, pos):
        if self.terrain_rects['desert'].collidepoint(pos):
            return "desert"
        if self.terrain_rects['water'].collidepoint(pos):
            return "water"
        return "grass"

    def _update_creatures(self):
        # Apply forces and update positions
        for i, creature in enumerate(self.creatures):
            # Repulsion from other creatures
            repulsion_force = pygame.Vector2(0, 0)
            for j, other in enumerate(self.creatures):
                if i == j: continue
                vec_to_other = other['pos'] - creature['pos']
                dist = vec_to_other.length()
                if 0 < dist < self.CREATURE_REPULSION_RADIUS:
                    repulsion_force -= vec_to_other.normalize() * (self.CREATURE_REPULSION / (dist + 1))
            creature['vel'] += repulsion_force

            # Apply terrain-based speed modifiers and drag
            terrain = self._get_terrain_at_pos(creature['pos'])
            speed_mod = 1.0
            if creature['type'] == "swimmer":
                speed_mod = 2.0 if terrain == "water" else 0.2
            elif creature['type'] == "crawler":
                speed_mod = 0.3 if terrain == "water" else 1.0
            
            creature['vel'] *= self.CREATURE_DRAG * speed_mod

            # Update position
            creature['pos'] += creature['vel']

            # Boundary collision
            creature['pos'].x = np.clip(creature['pos'].x, creature['radius'], self.WIDTH - creature['radius'])
            creature['pos'].y = np.clip(creature['pos'].y, creature['radius'], self.HEIGHT - creature['radius'])

            # Update trail
            creature['trail'].append(tuple(creature['pos']))
            if len(creature['trail']) > 15:
                creature['trail'].pop(0)

            # Update transform flash effect
            if creature['transform_flash'] > 0:
                creature['transform_flash'] -= 1

    def _check_resource_collection(self):
        collected_count = 0
        for res in self.resources:
            if not res['collected']:
                for creature in self.creatures:
                    dist = creature['pos'].distance_to(res['pos'])
                    if dist < creature['radius'] + res['radius']:
                        res['collected'] = True
                        self.score += 1
                        collected_count += 1
                        break # a resource can only be collected once
        return collected_count

    def _update_transformation(self):
        self.transformation_timer -= 1
        if self.transformation_timer <= 0:
            self.transformation_timer = self.TRANSFORM_INTERVAL
            for creature in self.creatures:
                creature['type'] = random.choice(["flyer", "swimmer", "crawler"])
                creature['transform_flash'] = 15 # frames
            # sfx: transform_all.wav

    def _get_dist_to_nearest_resource(self, pos):
        uncollected = [r['pos'] for r in self.resources if not r['collected']]
        if not uncollected:
            return 0
        distances = [pos.distance_to(r_pos) for r_pos in uncollected]
        return min(distances)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Render Terrain
        pygame.draw.rect(self.screen, self.COLOR_GRASS, self.terrain_rects['grass'])
        pygame.draw.rect(self.screen, self.COLOR_WATER, self.terrain_rects['water'])
        pygame.draw.rect(self.screen, self.COLOR_DESERT, self.terrain_rects['desert'])

        # Render Resources
        for res in self.resources:
            if not res['collected']:
                pos = (int(res['pos'].x), int(res['pos'].y))
                radius = int(res['radius'])
                # Pulsing glow effect
                res['glow'] += res['glow_dir'] * 0.05
                if res['glow'] > 1.0 or res['glow'] < 0.5:
                    res['glow_dir'] *= -1
                
                glow_alpha = int(100 * res['glow'])
                glow_color = (*self.COLOR_RESOURCE[:3], glow_alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 3, glow_color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 3, glow_color)

                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_RESOURCE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_RESOURCE)

        # Render Creatures
        for i, creature in enumerate(self.creatures):
            pos_v = creature['pos']
            pos = (int(pos_v.x), int(pos_v.y))
            radius = creature['radius']

            # Trail
            if len(creature['trail']) > 1:
                trail_color = (*creature['color'], 100)
                pygame.draw.aalines(self.screen, trail_color, False, creature['trail'], 1)

            # Selected Creature Glow
            if i == self.selected_creature_idx:
                glow_radius = int(radius * 1.8)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_SELECT_GLOW, 50), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

            # Transformation Flash
            if creature['transform_flash'] > 0:
                flash_alpha = int(255 * (creature['transform_flash'] / 15))
                flash_color = (*self.COLOR_SELECT_GLOW[:3], flash_alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 5, flash_color)

            # Creature Body
            self._draw_creature_shape(self.screen, creature)

    def _draw_creature_shape(self, surface, creature):
        pos = creature['pos']
        radius = creature['radius']
        color = creature['color']
        
        points = []
        if creature['type'] == 'flyer': # Triangle
            angle = creature['vel'].angle_to(pygame.Vector2(1,0)) if creature['vel'].length() > 0.1 else 0
            for i in range(3):
                a = math.radians(angle + 120 * i)
                points.append((pos.x + math.cos(a) * radius, pos.y - math.sin(a) * radius))
        elif creature['type'] == 'swimmer': # Elongated Hexagon
            angle = creature['vel'].angle_to(pygame.Vector2(1,0)) if creature['vel'].length() > 0.1 else 0
            rads = [math.radians(angle + a) for a in [0, 60, 120, 180, 240, 300]]
            scales = [1.5, 0.8, 0.8, 1.5, 0.8, 0.8]
            for i in range(6):
                points.append((pos.x + math.cos(rads[i]) * radius * scales[i], pos.y - math.sin(rads[i]) * radius * scales[i]))
        else: # 'crawler', Pentagon
            angle = 0 # Crawlers don't orient
            for i in range(5):
                a = math.radians(angle + 72 * i + 90)
                points.append((pos.x + math.cos(a) * radius, pos.y - math.sin(a) * radius))

        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surface, int_points, color)
        pygame.gfxdraw.filled_polygon(surface, int_points, color)


    def _render_ui(self):
        # --- Resource Bar ---
        bar_x, bar_y, bar_w, bar_h = 10, 10, 200, 20
        progress = min(1.0, self.score / self.WIN_SCORE)
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, (100, 200, 255), (bar_x, bar_y, int(bar_w * progress), bar_h))
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 2)
        score_text = self.font_ui.render(f"Resources: {self.score}/{self.WIN_SCORE}", True, (255, 255, 255))
        self.screen.blit(score_text, (bar_x + bar_w + 10, bar_y))

        # --- Transformation Timer Bar ---
        trans_bar_x, trans_bar_y, trans_bar_w, trans_bar_h = 10, 40, 200, 10
        trans_progress = self.transformation_timer / self.TRANSFORM_INTERVAL
        pygame.draw.rect(self.screen, (50, 50, 50), (trans_bar_x, trans_bar_y, trans_bar_w, trans_bar_h))
        pygame.draw.rect(self.screen, (255, 150, 50), (trans_bar_x, trans_bar_y, int(trans_bar_w * trans_progress), trans_bar_h))
        trans_text = self.font_ui.render("Transform In:", True, (255, 255, 255))
        self.screen.blit(trans_text, (trans_bar_x + trans_bar_w + 10, trans_bar_y - 5))

        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.score >= self.WIN_SCORE else "TIME UP"
            end_text = self.font_big.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to remove the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Creature Migration")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # --- Keyboard Mapping ---
    # ARROWS: Move
    # SPACE:  Select next creature
    
    print("\n--- Manual Control ---")
    print("ARROWS: Move selected creature")
    print("SPACE:  Select next creature")
    print("R:      Reset environment")
    print("----------------------")

    while not done:
        # Default actions
        movement_action = 0  # 0=none
        space_action = 0     # 0=released
        shift_action = 0     # 0=released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"Environment reset. Score: {info['score']}")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Reset on finish to allow continued play
            obs, info = env.reset()
            total_reward = 0

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()