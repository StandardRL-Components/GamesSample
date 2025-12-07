import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Hold Space for a long jump, or Shift for a medium jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated neon platforms, dodging obstacles to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 10000
    MAX_STEPS = 3000  # Increased for a reasonable play time at 30fps
    FPS = 30

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (30, 0, 50)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_TEXT = (255, 255, 255)
    PLATFORM_PALETTE = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 128),  # Spring Green
    ]
    FINISH_LINE_COLOR_1 = (255, 255, 255)
    FINISH_LINE_COLOR_2 = (180, 180, 180)

    # Physics
    GRAVITY = 0.8
    LONG_JUMP_POWER = -15
    MEDIUM_JUMP_POWER = -12
    WALK_SPEED = 6
    AIR_CONTROL = 0.5
    FRICTION = -0.12
    DAMPING = 0.98

    # Player
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 30
    INITIAL_LIVES = 3

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        self.render_mode = render_mode
        self.np_random = None

        # State variables are initialized in reset()
        self.player = {}
        self.platforms = []
        self.obstacles = []
        self.particles = []
        self.camera = {}
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.last_action = [0, 0, 0]
        self.furthest_platform_generated = 0
        self.finish_line_x = self.WORLD_WIDTH - 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.last_action = [0, 0, 0]
        self.furthest_platform_generated = 0

        self._initialize_player()
        self._initialize_camera()  # FIX: Initialize camera before level
        self._initialize_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False
        truncated = False

        if not self.game_over:
            prev_x = self.player["pos"][0]
            self._update_player(action)
            self._update_world()
            reward += self._handle_collisions()

            # Positional reward for moving right
            dx = self.player["pos"][0] - prev_x
            if dx > 0:
                reward += dx * 0.01
            else:
                reward += dx * 0.02  # Slightly higher penalty for moving left

        self.score += reward
        self.steps += 1

        if self.lives <= 0:
            self.game_over = True
            terminated = True
            reward -= 50 # Penalty for losing all lives

        if self.player["pos"][0] >= self.finish_line_x:
            self.game_over = True
            terminated = True
            reward += 100  # Goal reward
            self.score += 100

        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.last_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _initialize_player(self):
        self.player = {
            "pos": np.array([100.0, 200.0]),
            "vel": np.array([0.0, 0.0]),
            "on_platform": False,
            "last_safe_platform_idx": 0,
            "anim_state": {"squash": 0, "stretch": 0},  # Frames remaining for animation
        }

    def _initialize_level(self):
        self.platforms = []
        self.obstacles = []
        self.particles = []

        # Starting platform
        start_platform = pygame.Rect(50, 250, 200, 20)
        self.platforms.append({"rect": start_platform, "color": self.PLATFORM_PALETTE[0]})
        self.player["pos"] = np.array([start_platform.centerx, start_platform.top - self.PLAYER_HEIGHT / 2], dtype=float)
        self.player["on_platform"] = True

        self.furthest_platform_generated = start_platform.right
        self._generate_level_chunks()

        # Final platform
        finish_platform = pygame.Rect(self.finish_line_x, 200, 200, 20)
        self.platforms.append({"rect": finish_platform, "color": self.PLATFORM_PALETTE[1]})

    def _initialize_camera(self):
        self.camera = {
            "pos": np.array([0.0, 0.0]),
            "shake": 0
        }
        # This might run before player pos is finalized on a platform, but it's a good first guess
        self.camera["pos"][0] = self.player["pos"][0] - self.SCREEN_WIDTH / 2
        self.camera["pos"][1] = self.player["pos"][1] - self.SCREEN_HEIGHT / 2

    def _generate_level_chunks(self):
        while self.furthest_platform_generated < self.camera["pos"][0] + self.SCREEN_WIDTH * 1.5:
            if self.furthest_platform_generated > self.finish_line_x - 500:  # Stop generating near the end
                break

            last_platform = self.platforms[-1]["rect"]

            # Platform generation
            gap_x = self.np_random.integers(80, 180)
            gap_y = self.np_random.integers(-80, 80)
            width = self.np_random.integers(80, 200)
            height = 20

            new_x = last_platform.right + gap_x
            new_y = np.clip(last_platform.y + gap_y, 100, self.SCREEN_HEIGHT - 50)

            new_platform_rect = pygame.Rect(new_x, new_y, width, height)
            new_platform_color_index = self.np_random.integers(0, len(self.PLATFORM_PALETTE))
            new_platform_color = self.PLATFORM_PALETTE[new_platform_color_index]
            self.platforms.append({"rect": new_platform_rect, "color": new_platform_color})

            self.furthest_platform_generated = new_platform_rect.right

            # Obstacle generation (50% chance per platform)
            if self.np_random.random() < 0.5 and self.furthest_platform_generated > 500:
                obs_width = self.np_random.integers(20, 40)
                obs_height = self.np_random.integers(20, 40)
                obs_x = new_x - gap_x / 2 - obs_width / 2
                obs_y = new_y - self.np_random.integers(50, 150)

                # Difficulty scaling
                speed_scale = 1.0 + (self.steps / 500) * 0.05
                obs_speed = self.np_random.choice([-1, 1]) * self.np_random.uniform(1.5, 3.0) * speed_scale

                self.obstacles.append({
                    "rect": pygame.Rect(obs_x, obs_y, obs_width, obs_height),
                    "vel_x": obs_speed,
                    "initial_y": obs_y
                })

    def _update_player(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        last_space_held, last_shift_held = self.last_action[1] == 1, self.last_action[2] == 1

        # --- Horizontal Movement ---
        f_x = 0
        if movement == 3:  # Left
            f_x = -self.WALK_SPEED
        elif movement == 4:  # Right
            f_x = self.WALK_SPEED
        
        if self.player["on_platform"]:
            # Apply friction
            friction = self.player["vel"][0] * self.FRICTION
            self.player["vel"][0] += friction
            self.player["vel"][0] += f_x
        else:  # Air control
            self.player["vel"][0] += f_x * self.AIR_CONTROL

        # --- Jumping ---
        if self.player["on_platform"]:
            self.player["vel"][1] = 0
            if space_held and not last_space_held:  # Long jump on new press
                self.player["vel"][1] = self.LONG_JUMP_POWER
                self.player["on_platform"] = False
                self._create_particles(self.player["pos"] + np.array([0, self.PLAYER_HEIGHT / 2]), 20, self.COLOR_PLAYER)
                self.player["anim_state"]["stretch"] = 8  # Stretch for 8 frames
            elif shift_held and not last_shift_held:  # Medium jump
                self.player["vel"][1] = self.MEDIUM_JUMP_POWER
                self.player["on_platform"] = False
                self._create_particles(self.player["pos"] + np.array([0, self.PLAYER_HEIGHT / 2]), 15, self.PLATFORM_PALETTE[2])
                self.player["anim_state"]["stretch"] = 6

        # --- Physics Update ---
        if not self.player["on_platform"]:
            self.player["vel"][1] += self.GRAVITY

        self.player["vel"] *= self.DAMPING
        self.player["pos"] += self.player["vel"]

        # Update animation states
        for key in self.player["anim_state"]:
            self.player["anim_state"][key] = max(0, self.player["anim_state"][key] - 1)

    def _update_world(self):
        # Update obstacles
        for obs in self.obstacles:
            obs["rect"].x += obs["vel_x"]
            obs["rect"].y = obs["initial_y"] + math.sin(self.steps * 0.05 + obs["rect"].x * 0.01) * 20
            if not 0 < obs["rect"].centerx < self.WORLD_WIDTH:
                obs["vel_x"] *= -1

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"][1] += self.GRAVITY * 0.1
            p["life"] -= 1

        # Update camera
        target_cam_x = self.player["pos"][0] - self.SCREEN_WIDTH * 0.5
        target_cam_y = self.player["pos"][1] - self.SCREEN_HEIGHT * 0.6
        self.camera["pos"][0] += (target_cam_x - self.camera["pos"][0]) * 0.1
        self.camera["pos"][1] += (target_cam_y - self.camera["pos"][1]) * 0.05

        if self.camera["shake"] > 0:
            self.camera["shake"] -= 1

        self._generate_level_chunks()

    def _handle_collisions(self):
        reward = 0
        player_rect = self._get_player_rect()
        
        self.player["on_platform"] = False
        if self.player["vel"][1] >= 0:
            for i, p_data in enumerate(self.platforms):
                platform_rect = p_data["rect"]
                if player_rect.colliderect(platform_rect):
                    # Check if player was above the platform in the previous frame
                    prev_player_bottom = player_rect.bottom - self.player["vel"][1]
                    if prev_player_bottom <= platform_rect.top:
                        self.player["on_platform"] = True
                        self.player["vel"][1] = 0
                        self.player["pos"][1] = platform_rect.top - self.PLAYER_HEIGHT / 2
                        if self.player["last_safe_platform_idx"] != i:
                            reward += 5 # Reward for reaching a new platform
                            self.player["last_safe_platform_idx"] = i
                            self._create_particles(self.player["pos"] + np.array([0, self.PLAYER_HEIGHT / 2]), 10, p_data["color"])
                            self.player["anim_state"]["squash"] = 5
                            self.camera["shake"] = 3
                        break
        
        for obs in self.obstacles:
            if player_rect.colliderect(obs["rect"]):
                reward -= 10
                self._lose_life()
                break
        
        if player_rect.top > self.camera["pos"][1] + self.SCREEN_HEIGHT + 50:
            self._lose_life()

        return reward

    def _lose_life(self):
        self.lives -= 1
        self.camera["shake"] = 15
        self._create_particles(self.player["pos"], 50, self.COLOR_OBSTACLE)
        if self.lives > 0:
            safe_platform = self.platforms[self.player["last_safe_platform_idx"]]["rect"]
            self.player["pos"] = np.array([safe_platform.centerx, safe_platform.top - self.PLAYER_HEIGHT / 2], dtype=float)
            self.player["vel"] = np.array([0.0, 0.0])
            self.player["on_platform"] = True
        else:
            self.game_over = True

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_pos": tuple(self.player["pos"]),
            "player_vel": tuple(self.player["vel"]),
        }

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            color = [
                np.interp(y, [0, self.SCREEN_HEIGHT], [self.COLOR_BG_TOP[i], self.COLOR_BG_BOTTOM[i]])
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        cam_x, cam_y = self.camera["pos"]
        if self.camera["shake"] > 0:
            cam_x += self.np_random.uniform(-5, 5)
            cam_y += self.np_random.uniform(-5, 5)

        finish_rect = pygame.Rect(self.finish_line_x, 0, 20, self.WORLD_WIDTH)
        if finish_rect.colliderect(pygame.Rect(cam_x, cam_y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)):
            check_size = 20
            for y_tile in range(int(cam_y // check_size), int((cam_y + self.SCREEN_HEIGHT) // check_size) + 1):
                color = self.FINISH_LINE_COLOR_1 if y_tile % 2 == 0 else self.FINISH_LINE_COLOR_2
                screen_pos = (int(self.finish_line_x - cam_x), int(y_tile * check_size - cam_y))
                pygame.draw.rect(self.screen, color, (*screen_pos, check_size, check_size))

        for p_data in self.platforms:
            rect = p_data["rect"]
            color = p_data["color"]
            screen_rect = pygame.Rect(rect.x - cam_x, rect.y - cam_y, rect.width, rect.height)
            pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)
            highlight_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.line(self.screen, highlight_color, screen_rect.topleft, screen_rect.topright, 2)

        for obs in self.obstacles:
            rect = obs["rect"]
            screen_rect = pygame.Rect(rect.x - cam_x, rect.y - cam_y, rect.width, rect.height)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect, border_radius=3)

        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            pos = (int(p["pos"][0] - cam_x), int(p["pos"][1] - cam_y))
            size = int(p["size"] * (p["life"] / p["max_life"]))
            if size > 0:
                s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p["color"], alpha), (size, size), size)
                self.screen.blit(s, (pos[0] - size, pos[1] - size))

        if self.lives > 0:
            self._render_player(cam_x, cam_y)

    def _render_player(self, cam_x, cam_y):
        player_rect = self._get_player_rect()

        squash = self.player["anim_state"]["squash"] / 5.0
        stretch = self.player["anim_state"]["stretch"] / 8.0

        width = self.PLAYER_WIDTH * (1 + squash * 0.4 - stretch * 0.3)
        height = self.PLAYER_HEIGHT * (1 - squash * 0.4 + stretch * 0.3)

        screen_pos_x = player_rect.centerx - cam_x
        screen_pos_y = player_rect.centery - cam_y

        draw_rect = pygame.Rect(0, 0, width, height)
        draw_rect.center = (int(screen_pos_x), int(screen_pos_y))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, draw_rect, border_radius=5)

        glow_surface = pygame.Surface((width * 2, height * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER, 30), glow_surface.get_rect(), border_radius=int(width * 0.5))
        self.screen.blit(glow_surface, (draw_rect.centerx - width, draw_rect.centery - height))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.player["pos"][0] >= self.finish_line_x else "GAME OVER"
            over_text = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_player_rect(self):
        return pygame.Rect(
            self.player["pos"][0] - self.PLAYER_WIDTH / 2,
            self.player["pos"][1] - self.PLAYER_HEIGHT / 2,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT,
        )

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()


# Example usage for human play
if __name__ == "__main__":
    # To run with display, comment out the os.environ line at the top
    # and instantiate with render_mode="human"
    # os.environ.pop("SDL_VIDEODRIVER", None)
    # env = GameEnv(render_mode="human")
    
    # Standard headless verification
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    # --- Verification ---
    # Test action space
    assert env.action_space.shape == (3,)
    assert env.action_space.nvec.tolist() == [5, 2, 2]
    
    # Test reset and observation space
    assert obs.shape == (env.SCREEN_HEIGHT, env.SCREEN_WIDTH, 3)
    assert obs.dtype == np.uint8
    assert isinstance(info, dict)
    
    # Test step
    test_action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(test_action)
    assert obs.shape == (env.SCREEN_HEIGHT, env.SCREEN_WIDTH, 3)
    assert isinstance(reward, (int, float))
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)
    print("✓ Implementation validated successfully")
    # --- End Verification ---

    # To run the game with a display window
    try:
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption(GameEnv.game_description)
        clock = pygame.time.Clock()

        obs, info = env.reset()
        running = True
        total_reward = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            movement = 0  # none
            if keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']:.2f}, Steps: {info['steps']}")
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
    except pygame.error as e:
        print("\nCould not run human-playable example; a display is likely not available.")
        print(f"Pygame error: {e}")
    finally:
        env.close()