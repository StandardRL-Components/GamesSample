
# Generated: 2025-08-28T00:00:50.621905
# Source Brief: brief_03656.md
# Brief Index: 3656

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑/Space to jump, ↓/Shift to duck. Your character runs automatically."
    )

    # User-facing description of the game
    game_description = (
        "Escape a procedurally generated cursed forest by dodging deadly traps. "
        "The longer you survive, the faster you run and the more frequent the traps become."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_TRAP = (220, 40, 40)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 1500
    WIN_DISTANCE = 5000
    DIFFICULTY_INTERVAL = 500  # Steps before increasing difficulty

    # Physics
    GRAVITY = 0.6
    JUMP_STRENGTH = -12
    INITIAL_SCROLL_SPEED = 4.0
    SPEED_INCREMENT = 0.2

    # Player
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT_RUN = 40
    PLAYER_HEIGHT_DUCK = 20
    PLAYER_X = WIDTH // 4  # Player is horizontally fixed
    GROUND_Y = HEIGHT - 60
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Initialize state attributes to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        self.player_y = 0
        self.player_vy = 0
        self.player_state = 'running' # 'running', 'jumping', 'ducking'
        self.player_rect = pygame.Rect(0,0,0,0)
        
        self.distance_traveled = 0.0
        self.world_scroll_speed = self.INITIAL_SCROLL_SPEED
        self.last_difficulty_increase = 0
        
        self.traps = []
        self.next_trap_distance = 0
        
        self.particles = []
        self.background_elements = []

        self.validate_implementation()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if seed is not provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()


        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.distance_traveled = 0.0
        self.world_scroll_speed = self.INITIAL_SCROLL_SPEED
        self.last_difficulty_increase = 0
        
        # Player state
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.player_state = 'running'
        self.player_rect = pygame.Rect(
            self.PLAYER_X, self.player_y - self.PLAYER_HEIGHT_RUN,
            self.PLAYER_WIDTH, self.PLAYER_HEIGHT_RUN
        )

        # World state
        self.traps = []
        self.particles = []
        self.next_trap_distance = self.WIDTH * 1.5
        
        # Procedurally generate background
        self.background_elements = []
        for _ in range(150):
            layer = self.np_random.integers(1, 5)
            self.background_elements.append({
                "x": self.np_random.integers(0, self.WIDTH),
                "y": self.np_random.integers(0, self.HEIGHT),
                "size": self.np_random.integers(1, 4) * layer,
                "layer": layer,
                "color": tuple(c * (0.4 + 0.1 * layer) for c in self.COLOR_BG)
            })
        self.background_elements.sort(key=lambda e: e['layer'])

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        self._handle_input(action)
        self._update_player()
        self._update_world()
        
        reward += 0.01 # Small reward for surviving a step
        
        collision, collision_reward = self._handle_traps()
        if collision:
            self.game_over = True
            reward += collision_reward
            # Sound: Player death sfx
        
        # Reward for dodging traps
        for trap in self.traps:
            if not trap.get("passed", False) and self.player_rect.left > trap["rect"].right:
                trap["passed"] = True
                reward += 1.0
                self.score += 10 # Add to visual score for dodging

        self._update_difficulty()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over: # Win condition
            reward += 100.0
            self.score += 1000
            
        if self.auto_advance:
            self.clock.tick(30)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        is_on_ground = self.player_y >= self.GROUND_Y
        
        # Prioritize jump over duck
        if (movement == 1 or space_held) and is_on_ground:
            self.player_vy = self.JUMP_STRENGTH
            self.player_state = 'jumping'
            self._spawn_particles(self.player_rect.midbottom, 10, self.COLOR_PLAYER)
            # Sound: Jump sfx
        elif movement == 2 or shift_held:
            if is_on_ground:
                self.player_state = 'ducking'
        else:
            if is_on_ground:
                self.player_state = 'running'

    def _update_player(self):
        # Apply gravity
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy

        # Ground constraint
        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            if self.player_state == 'jumping':
                self.player_state = 'running'
                self._spawn_particles(self.player_rect.midbottom, 15, self.COLOR_PLAYER)
                # Sound: Land sfx

        # Update player rect based on state
        height = self.PLAYER_HEIGHT_DUCK if self.player_state == 'ducking' else self.PLAYER_HEIGHT_RUN
        self.player_rect.height = height
        self.player_rect.bottom = self.player_y

    def _update_world(self):
        self.distance_traveled += self.world_scroll_speed
        
        # Scroll background
        for elem in self.background_elements:
            elem["x"] -= self.world_scroll_speed / elem["layer"]
            if elem["x"] + elem["size"] < 0:
                elem["x"] = self.WIDTH + self.np_random.integers(0, 50)
                elem["y"] = self.np_random.integers(0, self.HEIGHT)
        
        # Scroll and update traps
        for trap in self.traps:
            trap["rect"].x -= self.world_scroll_speed
            if trap["type"] == 'branch':
                if self.player_rect.centerx > trap["rect"].centerx - 50:
                    trap["vy"] += self.GRAVITY * 0.5
                    trap["rect"].y += trap["vy"]
            elif trap["type"] == 'blade':
                trap["angle"] += trap["speed"]
                offset = math.sin(trap["angle"]) * trap["range"]
                trap["rect"].centery = trap["origin_y"] + offset
            elif trap["type"] == 'log':
                trap["angle"] += 0.1 # visual rotation
        
        # Remove off-screen traps
        self.traps = [t for t in self.traps if t["rect"].right > 0]
        
        # Update particles
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _handle_traps(self):
        # Spawn new traps
        if self.distance_traveled > self.next_trap_distance:
            self._spawn_trap()
            min_dist = self.WIDTH * 0.6
            max_dist = self.WIDTH * 1.2
            self.next_trap_distance += self.np_random.uniform(min_dist, max_dist) / (self.world_scroll_speed / self.INITIAL_SCROLL_SPEED)

        # Check for collisions
        for trap in self.traps:
            if self.player_rect.colliderect(trap["rect"]):
                return True, -100.0 # Collision detected, return terminal reward
        return False, 0.0

    def _spawn_trap(self):
        trap_type = self.np_random.choice(['spike', 'pit', 'branch', 'blade', 'log'])
        x = self.WIDTH + 50
        
        if trap_type == 'spike':
            width = self.np_random.integers(40, 80)
            rect = pygame.Rect(x, self.GROUND_Y - 20, width, 20)
            self.traps.append({"type": "spike", "rect": rect})
        elif trap_type == 'pit':
            width = self.np_random.integers(60, 100)
            rect = pygame.Rect(x, self.GROUND_Y - 5, width, 20)
            self.traps.append({"type": "pit", "rect": rect})
        elif trap_type == 'branch':
            rect = pygame.Rect(x + self.np_random.integers(-30, 30), 0, 80, 20)
            self.traps.append({"type": "branch", "rect": rect, "vy": 0})
        elif trap_type == 'blade':
            origin_y = self.GROUND_Y - 100
            rect = pygame.Rect(x, origin_y, 15, 60)
            self.traps.append({
                "type": "blade", "rect": rect, "origin_y": origin_y,
                "angle": self.np_random.uniform(0, 2 * math.pi), 
                "speed": self.np_random.uniform(0.05, 0.1),
                "range": self.np_random.integers(60, 90)
            })
        elif trap_type == 'log':
            size = 50
            rect = pygame.Rect(x, self.GROUND_Y - size, size, size)
            self.traps.append({"type": "log", "rect": rect, "angle": 0})
        # Sound: Trap appears sfx (subtle)

    def _update_difficulty(self):
        if self.distance_traveled // self.DIFFICULTY_INTERVAL > self.last_difficulty_increase:
            self.last_difficulty_increase += 1
            self.world_scroll_speed += self.SPEED_INCREMENT

    def _check_termination(self):
        return (
            self.game_over or
            self.steps >= self.MAX_STEPS or
            self.distance_traveled >= self.WIN_DISTANCE
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render background
        for elem in self.background_elements:
            pygame.draw.rect(self.screen, elem["color"], (elem["x"], elem["y"], elem["size"], elem["size"]))
        
        # Render ground
        pygame.draw.rect(self.screen, (10, 12, 18), (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Render traps
        for trap in self.traps:
            if trap["type"] == 'spike':
                points = [
                    trap["rect"].bottomleft,
                    (trap["rect"].centerx, trap["rect"].top),
                    trap["rect"].bottomright
                ]
                pygame.draw.polygon(self.screen, self.COLOR_TRAP, points)
            elif trap["type"] == 'pit':
                pygame.draw.rect(self.screen, (5, 8, 12), trap["rect"])
                pygame.draw.rect(self.screen, self.COLOR_TRAP, trap["rect"], 2)
            elif trap["type"] == 'branch':
                pygame.draw.rect(self.screen, (80, 50, 30), trap["rect"])
            elif trap["type"] == 'blade':
                pygame.draw.rect(self.screen, self.COLOR_TRAP, trap["rect"])
                pygame.draw.line(self.screen, (100,100,100), (trap["rect"].centerx, 0), trap["rect"].midtop, 2)
            elif trap["type"] == 'log':
                center = trap["rect"].center
                radius = trap["rect"].width // 2
                pygame.draw.circle(self.screen, (100, 70, 50), center, radius)
                # Lines to show rotation
                for i in range(4):
                    angle = trap["angle"] + i * math.pi / 2
                    start = (center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius)
                    end = (center[0] - math.cos(angle) * radius, center[1] - math.sin(angle) * radius)
                    pygame.draw.line(self.screen, (80, 50, 30), start, end, 3)

        # Render particles
        for p in self.particles:
            alpha = max(0, p["life"] * p["alpha_decay"])
            color = (*p["color"], alpha)
            pos = [int(p["pos"][0]), int(p["pos"][1])]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["life"] * 0.2), color)

        # Render player glow
        glow_radius = int(self.player_rect.height * 0.8)
        for i in range(glow_radius, 0, -2):
            alpha = 50 * (1 - i / glow_radius)
            pygame.gfxdraw.aacircle(self.screen, self.player_rect.centerx, self.player_rect.centery, i, (*self.COLOR_PLAYER_GLOW, alpha))
        
        # Render player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3)

    def _render_ui(self):
        dist_text = f"Distance: {int(self.distance_traveled):05d}"
        self._draw_text(dist_text, (10, 10), self.font_main)
        
        score_text = f"Score: {self.score}"
        self._draw_text(score_text, (10, 40), self.font_small)

        progress = min(1.0, self.distance_traveled / self.WIN_DISTANCE)
        bar_width = self.WIDTH - 20
        pygame.draw.rect(self.screen, (50, 50, 50), (10, self.HEIGHT - 20, bar_width, 10), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.HEIGHT - 20, bar_width * progress, 10), border_radius=3)

    def _draw_text(self, text, pos, font):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, pos)
        
    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 0)],
                "life": self.np_random.integers(20, 40),
                "color": color,
                "alpha_decay": 255 / 40
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.distance_traveled,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # --- Pygame setup for human play ---
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Cursed Forest Escape")
    clock = pygame.time.Clock()

    while running:
        # --- Action mapping for human input ---
        keys = pygame.key.get_pressed()
        move = 0
        if keys[pygame.K_UP]: move = 1
        elif keys[pygame.K_DOWN]: move = 2
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to the display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            print("Press 'R' to restart.")

        clock.tick(30) # Limit frame rate for human play

    env.close()