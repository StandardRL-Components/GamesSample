import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:14:46.191736
# Source Brief: brief_02857.md
# Brief Index: 2857
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Race a seed across a procedurally generated field, riding rhythmic gusts of wind
    and crafting protective shells to reach the finish line.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Race a seed across a procedurally generated field, riding rhythmic gusts of wind "
        "and crafting protective shells to reach the finish line."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to catch wind gusts for a speed boost."
    )
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FINISH_LINE_X = SCREEN_WIDTH - 40
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 23, 42)
    COLOR_PLAYER = (5, 255, 158)
    COLOR_PLAYER_GLOW = (5, 255, 158, 30)
    COLOR_OBSTACLE = (255, 59, 48)
    COLOR_DEBRIS = (255, 204, 0)
    COLOR_WIND = (52, 120, 246)
    COLOR_FINISH = (240, 240, 240)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_FRAME = (70, 80, 100)
    COLOR_HEALTH = (46, 204, 113)
    COLOR_SHELL = (0, 191, 255)
    COLOR_COMBO = (253, 184, 19)

    # Player
    PLAYER_RADIUS = 8
    PLAYER_SPEED = 4.0
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_SHELL = 100

    # Wind
    WIND_CYCLE_TIME = 120  # frames
    WIND_CATCH_WINDOW = 20 # frames
    WIND_BOOST_STRENGTH = 6.0

    # Obstacles
    OBSTACLE_SIZE = 15
    OBSTACLE_BASE_SPEED = 1.0
    OBSTACLE_SPAWN_INTERVAL = 40 # frames
    OBSTACLE_DAMAGE = 25

    # Debris
    DEBRIS_SIZE = 8
    DEBRIS_HEAL_AMOUNT = 20
    MAX_DEBRIS = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables are initialized in reset()
        self.player = {}
        self.obstacles = []
        self.debris = []
        self.particles = []
        self.wind_lines = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wind_timer = 0
        self.wind_direction = pygame.Vector2(0, 0)
        self.obstacle_spawn_timer = 0
        self.obstacle_current_speed = self.OBSTACLE_BASE_SPEED
        self.combo = 0
        self.space_was_held = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.combo = 0
        self.space_was_held = False

        self.player = {
            "pos": pygame.Vector2(50, self.SCREEN_HEIGHT / 2),
            "health": self.PLAYER_MAX_HEALTH,
            "shell": 0,
            "boost_timer": 0,
            "boost_dir": pygame.Vector2(0, 0)
        }
        
        self.obstacles.clear()
        self.debris.clear()
        self.particles.clear()
        
        self.obstacle_spawn_timer = 0
        self.obstacle_current_speed = self.OBSTACLE_BASE_SPEED
        self._spawn_initial_debris()

        self.wind_timer = self.WIND_CYCLE_TIME
        self._update_wind_direction()
        self._init_wind_lines()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        prev_player_x = self.player["pos"].x

        # --- Action Handling ---
        movement_action = action[0]
        space_held = action[1] == 1
        # shift_held is unused per brief
        
        space_pressed_this_frame = space_held and not self.space_was_held
        self.space_was_held = space_held
        
        # --- Update Game Logic ---
        self._update_player(movement_action, space_pressed_this_frame)
        reward += self._update_wind(space_pressed_this_frame)
        reward += self._update_obstacles()
        reward += self._update_debris()
        self._update_particles()

        # Positional reward
        reward += (self.player["pos"].x - prev_player_x) * 0.01

        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True
            if terminated and self.player["health"] <= 0:
                 self.score -= 50 # Penalty for losing
            if terminated and self.player["pos"].x >= self.FINISH_LINE_X:
                 self.score += 100 # Bonus for winning


        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement, space_pressed):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player["pos"] += move_vec * self.PLAYER_SPEED

        # Apply wind boost
        if self.player["boost_timer"] > 0:
            self.player["pos"] += self.player["boost_dir"] * self.WIND_BOOST_STRENGTH
            self.player["boost_timer"] -= 1

        # Boundary checks
        self.player["pos"].x = max(self.PLAYER_RADIUS, min(self.player["pos"].x, self.SCREEN_WIDTH - self.PLAYER_RADIUS))
        self.player["pos"].y = max(self.PLAYER_RADIUS, min(self.player["pos"].y, self.SCREEN_HEIGHT - self.PLAYER_RADIUS))

    def _update_wind(self, space_pressed):
        reward = 0
        self.wind_timer -= 1

        is_in_catch_window = self.wind_timer <= self.WIND_CATCH_WINDOW and self.wind_timer > 0

        if space_pressed:
            if is_in_catch_window:
                # Successful catch
                self.player["boost_timer"] = 15 # frames of boost
                self.player["boost_dir"] = self.wind_direction.copy()
                self.combo += 1
                reward += 1.0
                # sfx: wind_whoosh_success.wav
                self._create_particles(self.player["pos"], 20, self.COLOR_WIND, 3, 20)
                if self.combo > 0 and self.combo % 5 == 0:
                    reward += 5.0 # Combo bonus
            else:
                # Missed catch, reset combo
                self.combo = 0
                # sfx: buzz_error.wav
                
        if self.wind_timer <= 0:
            self.wind_timer = self.WIND_CYCLE_TIME
            self._update_wind_direction()

        # Update visual wind lines
        for line in self.wind_lines:
            line["pos"] += line["vel"]
            if line["pos"].x < -20 or line["pos"].x > self.SCREEN_WIDTH + 20 or \
               line["pos"].y < -20 or line["pos"].y > self.SCREEN_HEIGHT + 20:
                line["pos"] = self._get_random_edge_pos()
                line["vel"] = self.wind_direction * line["speed"]

        return reward

    def _update_obstacles(self):
        reward = 0
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_current_speed += 0.05

        # Spawn new obstacles
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self.obstacle_spawn_timer = self.OBSTACLE_SPAWN_INTERVAL
            side = random.choice([0, 1, 2, 3]) # 0:top, 1:bottom, 2:left, 3:right
            if side == 0:
                pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), -self.OBSTACLE_SIZE)
                vel = pygame.Vector2(random.uniform(-0.5, 0.5), 1)
            elif side == 1:
                pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.OBSTACLE_SIZE)
                vel = pygame.Vector2(random.uniform(-0.5, 0.5), -1)
            elif side == 2:
                pos = pygame.Vector2(-self.OBSTACLE_SIZE, random.uniform(0, self.SCREEN_HEIGHT))
                vel = pygame.Vector2(1, random.uniform(-0.5, 0.5))
            else: # side == 3
                pos = pygame.Vector2(self.SCREEN_WIDTH + self.OBSTACLE_SIZE, random.uniform(0, self.SCREEN_HEIGHT))
                vel = pygame.Vector2(-1, random.uniform(-0.5, 0.5))
            
            vel.normalize_ip()
            vel *= self.obstacle_current_speed
            self.obstacles.append({"pos": pos, "vel": vel})

        # Move and check collisions
        for obs in list(self.obstacles):
            obs["pos"] += obs["vel"]
            # Collision with player
            if obs["pos"].distance_to(self.player["pos"]) < self.OBSTACLE_SIZE / 2 + self.PLAYER_RADIUS:
                self.obstacles.remove(obs)
                self._handle_damage(self.OBSTACLE_DAMAGE)
                reward -= 2.0
                # sfx: player_hit.wav
                continue
            # Remove if off-screen
            if not pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT).colliderect(
                pygame.Rect(obs["pos"].x - self.OBSTACLE_SIZE/2, obs["pos"].y - self.OBSTACLE_SIZE/2, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
            ):
                 # Give a small buffer before removing
                if obs["pos"].x < -50 or obs["pos"].x > self.SCREEN_WIDTH + 50 or \
                   obs["pos"].y < -50 or obs["pos"].y > self.SCREEN_HEIGHT + 50:
                    self.obstacles.remove(obs)
        return reward

    def _handle_damage(self, amount):
        self._create_particles(self.player["pos"], 30, self.COLOR_OBSTACLE, 4, 25)
        self.combo = 0 # Getting hit breaks combo
        
        shell_damage = min(self.player["shell"], amount)
        self.player["shell"] -= shell_damage
        
        health_damage = amount - shell_damage
        self.player["health"] -= health_damage
        self.player["health"] = max(0, self.player["health"])
        
        if self.player["health"] <= 0:
            self.game_over = True

    def _update_debris(self):
        reward = 0
        for deb in list(self.debris):
            if deb["pos"].distance_to(self.player["pos"]) < self.DEBRIS_SIZE + self.PLAYER_RADIUS:
                self.debris.remove(deb)
                self.player["shell"] = min(self.PLAYER_MAX_SHELL, self.player["shell"] + self.DEBRIS_HEAL_AMOUNT)
                reward += 0.5
                self._create_particles(self.player["pos"], 15, self.COLOR_DEBRIS, 2, 15)
                # sfx: collect_debris.wav
        return reward

    def _update_particles(self):
        for p in list(self.particles):
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.player["health"] <= 0:
            return True
        if self.player["pos"].x >= self.FINISH_LINE_X:
            return True
        return False

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
            "health": self.player.get("health", 0),
            "shell": self.player.get("shell", 0),
            "combo": self.combo,
            "pos_x": self.player.get("pos", pygame.Vector2(0,0)).x,
            "pos_y": self.player.get("pos", pygame.Vector2(0,0)).y
        }

    def _render_game(self):
        self._render_wind_lines()
        self._render_finish_line()
        self._render_debris()
        self._render_obstacles()
        self._render_player()
        self._render_particles()

    def _render_wind_lines(self):
        for line in self.wind_lines:
            start_pos = line["pos"]
            end_pos = line["pos"] - line["vel"] * 2
            alpha = int(max(0, min(255, 100 * (self.wind_timer / self.WIND_CYCLE_TIME))))
            if alpha > 10:
                pygame.draw.aaline(self.screen, (*self.COLOR_WIND, alpha), start_pos, end_pos, 1)

    def _render_finish_line(self):
        for y in range(0, self.SCREEN_HEIGHT, 20):
            color = self.COLOR_FINISH if (y // 20) % 2 == 0 else self.COLOR_BG
            pygame.draw.line(self.screen, color, (self.FINISH_LINE_X, y), (self.FINISH_LINE_X, y + 10), 5)

    def _render_debris(self):
        for deb in self.debris:
            p = deb["pos"]
            size = self.DEBRIS_SIZE
            points = [
                (p.x, p.y - size),
                (p.x - size * 0.866, p.y + size * 0.5),
                (p.x + size * 0.866, p.y + size * 0.5)
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_DEBRIS)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_DEBRIS)

    def _render_obstacles(self):
        for obs in self.obstacles:
            rect = pygame.Rect(obs["pos"].x - self.OBSTACLE_SIZE / 2, obs["pos"].y - self.OBSTACLE_SIZE / 2, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)

    def _render_player(self):
        pos_int = (int(self.player["pos"].x), int(self.player["pos"].y))
        
        # Glow effect
        for i in range(self.PLAYER_RADIUS, self.PLAYER_RADIUS + 10, 2):
            alpha = 50 * (1 - (i - self.PLAYER_RADIUS) / 10)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], i, (*self.COLOR_PLAYER[:3], int(alpha)))
            
        # Player core
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        
        # Player UI (Health, Shell, Combo)
        bar_width = 40
        bar_height = 5
        # Health Bar
        health_pct = self.player["health"] / self.PLAYER_MAX_HEALTH
        health_bar_y = pos_int[1] - self.PLAYER_RADIUS - 15
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, (pos_int[0] - bar_width/2, health_bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (pos_int[0] - bar_width/2, health_bar_y, bar_width * health_pct, bar_height))
        
        # Shell Bar
        if self.player["shell"] > 0:
            shell_pct = self.player["shell"] / self.PLAYER_MAX_SHELL
            shell_bar_y = health_bar_y + bar_height + 2
            pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, (pos_int[0] - bar_width/2, shell_bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_SHELL, (pos_int[0] - bar_width/2, shell_bar_y, bar_width * shell_pct, bar_height))
        
        # Combo Meter
        if self.combo > 0:
            combo_text = self.font_small.render(f"x{self.combo}", True, self.COLOR_COMBO)
            text_rect = combo_text.get_rect(center=(pos_int[0], pos_int[1] + self.PLAYER_RADIUS + 15))
            self.screen.blit(combo_text, text_rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"][:3], alpha)
            if alpha > 0:
                pygame.draw.rect(self.screen, color, (int(p["pos"].x), int(p["pos"].y), p["size"], p["size"]))

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 30))

        # Wind Indicator
        ui_center_x = self.SCREEN_WIDTH / 2
        
        # Wind timer bar
        is_in_catch_window = self.wind_timer <= self.WIND_CATCH_WINDOW and self.wind_timer > 0
        bar_color = self.COLOR_COMBO if is_in_catch_window else self.COLOR_WIND
        bar_width = 150
        bar_progress = max(0, self.wind_timer - self.WIND_CATCH_WINDOW) / (self.WIND_CYCLE_TIME - self.WIND_CATCH_WINDOW)
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, (ui_center_x - bar_width / 2, self.SCREEN_HEIGHT - 30, bar_width, 10))
        pygame.draw.rect(self.screen, bar_color, (ui_center_x - bar_width / 2, self.SCREEN_HEIGHT - 30, bar_width * bar_progress, 10))
        
        # Catch window indicator
        catch_width = bar_width * (self.WIND_CATCH_WINDOW / self.WIND_CYCLE_TIME)
        catch_x = ui_center_x - bar_width / 2
        pygame.draw.rect(self.screen, self.COLOR_COMBO, (catch_x, self.SCREEN_HEIGHT - 30, catch_width, 10), 2)
        
        # Wind direction arrow
        arrow_center = pygame.Vector2(ui_center_x, self.SCREEN_HEIGHT - 50)
        arrow_end = arrow_center + self.wind_direction * 20
        self._draw_arrow(self.screen, arrow_center, arrow_end, self.COLOR_WIND, 5, 10, 5)

    # --- Helper Methods ---
    def _spawn_initial_debris(self):
        for _ in range(self.MAX_DEBRIS):
            self.debris.append({
                "pos": pygame.Vector2(
                    random.uniform(100, self.FINISH_LINE_X - 50),
                    random.uniform(50, self.SCREEN_HEIGHT - 50)
                )
            })

    def _create_particles(self, pos, count, color, speed, lifetime):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, 1) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "color": color,
                "life": random.randint(lifetime // 2, lifetime),
                "max_life": lifetime,
                "size": random.randint(2, 4)
            })
    
    def _update_wind_direction(self):
        angle = random.uniform(0, 2 * math.pi)
        self.wind_direction = pygame.Vector2(math.cos(angle), math.sin(angle))
        for line in self.wind_lines:
            line["vel"] = self.wind_direction * line["speed"]
            
    def _init_wind_lines(self):
        self.wind_lines.clear()
        for _ in range(50):
            self.wind_lines.append({
                "pos": self._get_random_edge_pos(),
                "vel": self.wind_direction * random.uniform(1.0, 3.0),
                "speed": random.uniform(1.0, 3.0)
            })

    def _get_random_edge_pos(self):
        edge = random.randint(0, 3)
        if edge == 0: return pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), 0)
        if edge == 1: return pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT)
        if edge == 2: return pygame.Vector2(0, random.uniform(0, self.SCREEN_HEIGHT))
        return pygame.Vector2(self.SCREEN_WIDTH, random.uniform(0, self.SCREEN_HEIGHT))

    @staticmethod
    def _draw_arrow(surface, start, end, color, body_width, head_width, head_height):
        arrow = start - end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        
        pygame.draw.line(surface, color, start, end, body_width)
        
        # Create the arrow head
        p1 = end + pygame.Vector2(0, head_height/2).rotate(-angle)
        p2 = end + pygame.Vector2(head_width/2, -head_height/2).rotate(-angle)
        p3 = end + pygame.Vector2(-head_width/2, -head_height/2).rotate(-angle)
        pygame.draw.polygon(surface, color, (p1, p2, p3))


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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


if __name__ == "__main__":
    # --- Manual Play ---
    # The original code had a validation call in __init__, which is not standard.
    # It's better to run validation separately if needed.
    # We remove it from __init__ and call it here for demonstration.
    try:
        env = GameEnv(render_mode="rgb_array")
        # env.validate_implementation() # This was in the original __init__
    except Exception as e:
        print(f"Error during environment initialization or validation: {e}")
        exit()

    obs, info = env.reset()
    
    # Set up display for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc. depending on your OS
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Seed Racer")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")
    
    while not done:
        movement = 0 # no-op
        space = 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                    continue
        
        if done: break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, 0] # Shift is unused
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for reset key 'r' or quit key 'q'
            pass
        
        clock.tick(env.metadata["render_fps"])
        
    env.close()