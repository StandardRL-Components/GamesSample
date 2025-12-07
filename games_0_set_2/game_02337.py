
# Generated: 2025-08-28T04:34:53.830565
# Source Brief: brief_02337.md
# Brief Index: 2337

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    An escape-the-forest game where the player must navigate a procedurally
    generated world to reach an exit, while avoiding patrolling predators
    and managing a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to dash. Avoid the red predators and reach the light!"
    )

    # Short, user-facing description of the game
    game_description = (
        "Escape a dark, procedurally generated forest. Evade predators and reach the exit before time runs out."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FPS = 30 # For time calculations

        # Game constants
        self.WORLD_WIDTH = 2000
        self.MAX_TIME_SECONDS = 60
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        
        # Colors
        self.COLOR_BG = (20, 30, 20)
        self.COLOR_PLAYER = (100, 200, 255)
        self.COLOR_PLAYER_GLOW = (*self.COLOR_PLAYER, 50)
        self.COLOR_PLAYER_OUTLINE = (200, 255, 255)
        self.COLOR_PREDATOR = (255, 50, 50)
        self.COLOR_PREDATOR_OUTLINE = (255, 150, 150)
        self.COLOR_EXIT = (255, 255, 150)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (0, 0, 0)
        self.COLOR_PARTICLE = (200, 220, 255)

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.player_pos = None
        self.player_size = 20
        self.player_speed = 4.0
        self.player_dash_speed_bonus = 8.0
        self.last_horiz_direction = None
        self.prev_space_held = None
        self.dash_particles = None
        self.predators = None
        self.trees = None
        self.exit_pos = None
        self.exit_size = 40
        self.camera_x = None
        self.np_random = None
        
        # Call reset to initialize the game state
        self.reset()
        
        # Validate implementation after full initialization
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.prev_space_held = False
        self.dash_particles = []
        
        # Player state
        self.player_pos = [100.0, self.HEIGHT / 2.0]
        self.last_horiz_direction = 1

        # World state
        self.camera_x = 0
        self.exit_pos = [self.WORLD_WIDTH - 100, self.HEIGHT / 2]
        
        # Procedural generation
        self._generate_trees()
        self._generate_predators()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        dist_to_exit_before = math.hypot(self.player_pos[0] - self.exit_pos[0], self.player_pos[1] - self.exit_pos[1])
        
        # Determine movement vector
        move_vec = [0.0, 0.0]
        if movement == 1: move_vec[1] = -1.0 # Up
        elif movement == 2: move_vec[1] = 1.0  # Down
        elif movement == 3: move_vec[0] = -1.0 # Left
        elif movement == 4: move_vec[0] = 1.0  # Right

        if move_vec[0] != 0:
            self.last_horiz_direction = move_vec[0]
        
        # Apply base movement
        self.player_pos[0] += move_vec[0] * self.player_speed
        self.player_pos[1] += move_vec[1] * self.player_speed

        # Apply dash on space press (not hold)
        is_space_pressed = space_held and not self.prev_space_held
        if is_space_pressed:
            # Sound: Dash sfx
            self.player_pos[0] += self.last_horiz_direction * self.player_dash_speed_bonus
            self._create_dash_particles()
        self.prev_space_held = space_held

        # --- 2. Update Game State ---
        self.steps += 1
        self.time_remaining -= 1

        # Apply player boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_size / 2, self.WORLD_WIDTH - self.player_size / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_size / 2, self.HEIGHT - self.player_size / 2)

        self._update_particles()
        self._update_predators()
        self._update_camera()

        # --- 3. Calculate Reward & Check Termination ---
        reward = -0.01  # Time penalty
        terminated = False

        dist_to_exit_after = math.hypot(self.player_pos[0] - self.exit_pos[0], self.player_pos[1] - self.exit_pos[1])
        reward += (dist_to_exit_before - dist_to_exit_after) * 0.1 # Reward for getting closer

        # Check collisions
        player_rect = pygame.Rect(self.player_pos[0] - self.player_size / 2, self.player_pos[1] - self.player_size / 2, self.player_size, self.player_size)

        if self._check_predator_collision(player_rect):
            # Sound: Player death sfx
            reward = -10.0
            terminated = True
        elif self._check_exit_collision(player_rect):
            # Sound: Win sfx
            reward = 100.0
            terminated = True
        elif self.time_remaining <= 0:
            # Sound: Timeout sfx
            terminated = True

        if terminated:
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "time_remaining": round(self.time_remaining / self.FPS, 2),
            "player_pos": self.player_pos,
        }
        
    def close(self):
        pygame.quit()

    # --- Helper methods for game logic ---

    def _generate_trees(self):
        self.trees = []
        for _ in range(150):
            is_valid_pos = False
            while not is_valid_pos:
                x = self.np_random.uniform(0, self.WORLD_WIDTH)
                # Avoid player start and exit areas
                if not (50 < x < 200) and not (self.WORLD_WIDTH - 200 < x < self.WORLD_WIDTH - 50):
                    is_valid_pos = True
            
            depth = self.np_random.uniform(0.3, 1.0)
            self.trees.append({
                "x": x,
                "y": self.np_random.uniform(self.HEIGHT * 0.4, self.HEIGHT),
                "w": 10 * depth,
                "h": self.np_random.uniform(100, 300) * depth,
                "depth": depth,
                "color": tuple(int(c * depth * 0.7) for c in (40, 60, 40))
            })
        self.trees.sort(key=lambda t: t['depth'])

    def _generate_predators(self):
        self.predators = []
        # Slowest predator, as per brief
        slow_start_x = self.np_random.uniform(400, 600)
        self.predators.append(self._create_predator(
            start_x=slow_start_x,
            patrol_range=self.np_random.uniform(200, 400),
            speed=0.2 # 1 pixel per 5 frames at 30fps
        ))
        # Two faster predators
        for i in range(2):
            start_x = self.np_random.uniform(800 + i * 400, 1100 + i * 400)
            self.predators.append(self._create_predator(
                start_x=start_x,
                patrol_range=self.np_random.uniform(300, 500),
                speed=self.np_random.uniform(1.0, 2.5)
            ))

    def _create_predator(self, start_x, patrol_range, speed):
        return {
            "pos": [start_x, self.np_random.uniform(50, self.HEIGHT - 50)],
            "size": 25,
            "patrol_start": start_x,
            "patrol_end": start_x + patrol_range,
            "speed": speed,
            "direction": 1,
            "trail": []
        }

    def _update_particles(self):
        for p in self.dash_particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.dash_particles.remove(p)

    def _update_predators(self):
        for p in self.predators:
            p["pos"][0] += p["speed"] * p["direction"]
            if p["pos"][0] >= p["patrol_end"] or p["pos"][0] <= p["patrol_start"]:
                p["direction"] *= -1
            p["trail"].append(list(p["pos"]))
            if len(p["trail"]) > 20:
                p["trail"].pop(0)

    def _update_camera(self):
        target_camera_x = self.player_pos[0] - self.WIDTH / 2
        self.camera_x = max(0, min(target_camera_x, self.WORLD_WIDTH - self.WIDTH))

    def _create_dash_particles(self):
        for _ in range(20):
            self.dash_particles.append({
                "pos": [self.player_pos[0], self.player_pos[1]],
                "vel": [self.np_random.uniform(-3, 3) - self.last_horiz_direction * 5, self.np_random.uniform(-3, 3)],
                "life": 15,
            })

    def _check_predator_collision(self, player_rect):
        for p in self.predators:
            predator_rect = pygame.Rect(p["pos"][0] - p["size"] / 2, p["pos"][1] - p["size"] / 2, p["size"], p["size"])
            if player_rect.colliderect(predator_rect):
                return True
        return False

    def _check_exit_collision(self, player_rect):
        exit_rect = pygame.Rect(self.exit_pos[0] - self.exit_size / 2, self.exit_pos[1] - self.exit_size / 2, self.exit_size, self.exit_size)
        return player_rect.colliderect(exit_rect)

    # --- Helper methods for rendering ---

    def _render_game(self):
        self._render_parallax_background()
        self._render_exit()
        self._render_predators_and_trails()
        self._render_particles()
        self._render_player()

    def _render_parallax_background(self):
        for tree in self.trees:
            screen_x = (tree["x"] - self.camera_x * tree["depth"])
            if -tree["w"] < screen_x < self.WIDTH:
                pygame.draw.rect(self.screen, tree["color"], (screen_x, tree["y"] - tree["h"], tree["w"], tree["h"]))

    def _render_exit(self):
        exit_screen_pos = (int(self.exit_pos[0] - self.camera_x), int(self.exit_pos[1]))
        if -self.exit_size < exit_screen_pos[0] < self.WIDTH + self.exit_size:
            for i in range(self.exit_size, 0, -2):
                alpha = int(100 * (1 - i / self.exit_size))
                pygame.gfxdraw.filled_circle(self.screen, exit_screen_pos[0], exit_screen_pos[1], i, (*self.COLOR_EXIT, alpha))

    def _render_predators_and_trails(self):
        for p in self.predators:
            # Trails
            if len(p["trail"]) > 1:
                for i, pos in enumerate(p["trail"]):
                    alpha = int(80 * (i / len(p["trail"])))
                    radius = int(p["size"] / 3 * (i / len(p["trail"])))
                    trail_screen_pos = (int(pos[0] - self.camera_x), int(pos[1]))
                    pygame.gfxdraw.filled_circle(self.screen, trail_screen_pos[0], trail_screen_pos[1], radius, (*self.COLOR_PREDATOR, alpha))
            # Body
            pred_screen_pos = (int(p["pos"][0] - self.camera_x), int(p["pos"][1]))
            if -p["size"] < pred_screen_pos[0] < self.WIDTH + p["size"]:
                size = p["size"] + abs(math.sin(self.steps * 0.1)) * 5 # Pulsing animation
                rect = pygame.Rect(pred_screen_pos[0] - size/2, pred_screen_pos[1] - size/2, size, size)
                pygame.draw.ellipse(self.screen, self.COLOR_PREDATOR, rect)
                pygame.draw.ellipse(self.screen, self.COLOR_PREDATOR_OUTLINE, rect, 2)

    def _render_particles(self):
        for p in self.dash_particles:
            radius = int(p["life"] / 15 * 4)
            if radius > 0:
                particle_screen_pos = (int(p["pos"][0] - self.camera_x), int(p["pos"][1]))
                pygame.draw.circle(self.screen, self.COLOR_PARTICLE, particle_screen_pos, radius)

    def _render_player(self):
        player_screen_pos = (int(self.player_pos[0] - self.camera_x), int(self.player_pos[1]))
        bob = math.sin(self.steps * 0.2) * 2
        
        # Glow effect
        glow_size = self.player_size * 1.8
        glow_rect = pygame.Rect(player_screen_pos[0] - glow_size/2, player_screen_pos[1] - glow_size/2 + bob, glow_size, glow_size)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, glow_size, glow_size))
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Main body
        player_rect = pygame.Rect(player_screen_pos[0] - self.player_size/2, player_screen_pos[1] - self.player_size/2 + bob, self.player_size, self.player_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, 2, border_radius=3)

    def _render_ui(self):
        def draw_text_with_shadow(text, pos, font, color, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        time_text = f"TIME: {self.time_remaining / self.FPS:.1f}"
        draw_text_with_shadow(time_text, (10, 10), self.font_ui, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

        score_text = f"SCORE: {int(self.score)}"
        text_width = self.font_ui.size(score_text)[0]
        draw_text_with_shadow(score_text, (self.WIDTH - text_width - 10, 10), self.font_ui, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a visible window for testing
    import os
    os.environ.pop('SDL_VIDEODRIVER', None)

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for display
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Forest Escape")
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Key presses for manual control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(env.FPS)
        
    env.close()