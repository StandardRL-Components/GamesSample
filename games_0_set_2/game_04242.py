
# Generated: 2025-08-28T01:49:32.579289
# Source Brief: brief_04242.md
# Brief Index: 4242

        
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

    user_guide = (
        "Controls: Use ←→ to move, and ↑ or Space to jump. Collect yellow stars for points and reach the green finish line. Avoid red obstacles!"
    )

    game_description = (
        "A fast-paced neon platformer. Guide your robot through a hazardous, procedurally generated course. Jump over obstacles, collect bonus items, and race against the clock to reach the finish line."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.W, self.H = 640, 400
        self.FPS = 30 # As per auto_advance spec
        self.LEVEL_LENGTH_IN_SCREENS = 15
        self.LEVEL_LENGTH_PX = self.W * self.LEVEL_LENGTH_IN_SCREENS
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Physics
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -14
        self.PLAYER_SPEED = 7
        self.PLAYER_SIZE = 24
        self.FLOOR_Y = self.H - 40

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 128, 128)
        self.COLOR_OBSTACLE = (255, 0, 100)
        self.COLOR_OBSTACLE_GLOW = (128, 0, 50)
        self.COLOR_ITEM = (255, 255, 0)
        self.COLOR_ITEM_GLOW = (128, 128, 0)
        self.COLOR_FINISH = (0, 255, 100)
        self.COLOR_FINISH_GLOW = (0, 128, 50)
        self.COLOR_FLOOR = (50, 30, 80)
        self.COLOR_TEXT = (220, 220, 240)
        
        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Variables ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel_y = 0
        self.on_ground = False
        self.world_scroll_x = 0
        self.obstacles = []
        self.bonus_items = []
        self.particles = []
        self.bg_stars = []
        self.current_obstacle_speed = 0
        self.last_obstacle_x = 0
        self.last_item_x = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = [self.W * 0.2, self.FLOOR_Y - self.PLAYER_SIZE]
        self.player_vel_y = 0
        self.on_ground = True
        
        self.world_scroll_x = 0
        self.current_obstacle_speed = 4.0
        
        self.obstacles = []
        self.bonus_items = []
        self.particles = []
        self.bg_stars = []

        # Populate initial world
        for _ in range(150): # Background stars
            self.bg_stars.append({
                "pos": [self.np_random.integers(0, self.LEVEL_LENGTH_PX), self.np_random.integers(0, self.H)],
                "depth": self.np_random.uniform(0.1, 0.7)
            })
            
        self.last_obstacle_x = self.W * 1.5
        self.last_item_x = self.W * 1.2
        self._spawn_initial_entities()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01  # Small survival reward
        
        self._handle_input(action)
        self._update_player()
        self._update_world(action)
        self._update_entities()

        item_reward = self._handle_collisions()
        reward += item_reward

        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.player_pos[1] > self.H + 50: # Fell off world (shouldn't happen with floor)
            reward -= 100
            terminated = True
        if self.world_scroll_x >= self.LEVEL_LENGTH_PX: # Reached finish line
            reward += 100
            terminated = True
        if self.steps >= self.MAX_STEPS: # Timeout
            terminated = True
        
        # Collision termination is handled in _handle_collisions
        if self.game_over:
            reward -= 100
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Horizontal Movement
        if movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Jumping
        is_jumping = (movement == 1) or (space_held == 1)
        if is_jumping and self.on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            # SFX: Jump
            self._spawn_particles(self.player_pos[0] + self.PLAYER_SIZE/2, self.FLOOR_Y, 5, self.COLOR_FLOOR)

    def _update_player(self):
        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        # Floor collision
        if self.player_pos[1] >= self.FLOOR_Y - self.PLAYER_SIZE:
            self.player_pos[1] = self.FLOOR_Y - self.PLAYER_SIZE
            self.player_vel_y = 0
            self.on_ground = True
        
        # Screen bounds
        self.player_pos[0] = max(0, min(self.player_pos[0], self.W - self.PLAYER_SIZE))

    def _update_world(self, action):
        movement = action[0]
        # Scroll world if player moves right past center
        scroll_speed = 0
        if movement == 4 and self.player_pos[0] > self.W * 0.4:
            scroll_speed = self.PLAYER_SPEED * 0.8
            self.player_pos[0] -= scroll_speed
        
        self.world_scroll_x += scroll_speed
        
        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.current_obstacle_speed += 0.2

        # Move entities based on scroll
        for entity_list in [self.obstacles, self.bonus_items]:
            for entity in entity_list:
                entity["pos"][0] -= self.current_obstacle_speed + scroll_speed

    def _update_entities(self):
        # Update particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # Remove off-screen entities
        self.obstacles = [o for o in self.obstacles if o["pos"][0] > -o["size"]]
        self.bonus_items = [i for i in self.bonus_items if i["pos"][0] > -i["size"]]

        # Spawn new entities
        self._spawn_entities()

    def _spawn_initial_entities(self):
        while self.last_obstacle_x < self.LEVEL_LENGTH_PX - self.W:
            self._spawn_obstacle()
        while self.last_item_x < self.LEVEL_LENGTH_PX - self.W:
            self._spawn_item()

    def _spawn_entities(self):
        # Spawn new obstacles if the last one is on screen
        if self.obstacles and self.obstacles[-1]["pos"][0] < self.W:
            self._spawn_obstacle()

        # Spawn new items
        if self.bonus_items and self.bonus_items[-1]["pos"][0] < self.W:
            self._spawn_item()

    def _spawn_obstacle(self):
        if self.last_obstacle_x > self.LEVEL_LENGTH_PX - self.W:
            return
        size = self.np_random.integers(20, 40)
        obstacle_x = self.last_obstacle_x + self.np_random.integers(250, 500)
        self.obstacles.append({
            "pos": [obstacle_x, self.FLOOR_Y - size],
            "size": size,
            "type": "triangle"
        })
        self.last_obstacle_x = obstacle_x
    
    def _spawn_item(self):
        if self.last_item_x > self.LEVEL_LENGTH_PX - self.W:
            return
        size = 15
        item_x = self.last_item_x + self.np_random.integers(200, 600)
        item_y = self.FLOOR_Y - self.np_random.integers(40, 150)
        self.bonus_items.append({
            "pos": [item_x, item_y],
            "size": size
        })
        self.last_item_x = item_x

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        item_reward = 0

        for item in self.bonus_items[:]:
            item_rect = pygame.Rect(item["pos"][0], item["pos"][1], item["size"], item["size"])
            if player_rect.colliderect(item_rect):
                self.bonus_items.remove(item)
                item_reward += 10
                self.score += 10
                # SFX: Item collect
                self._spawn_particles(item["pos"][0], item["pos"][1], 20, self.COLOR_ITEM)
        
        for obstacle in self.obstacles:
            obs_rect = pygame.Rect(obstacle["pos"][0], obstacle["pos"][1], obstacle["size"], obstacle["size"])
            if player_rect.colliderect(obs_rect):
                self.game_over = True
                # SFX: Explosion
                self._spawn_particles(self.player_pos[0], self.player_pos[1], 50, self.COLOR_OBSTACLE)
                break
        
        return item_reward

    def _spawn_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw parallax background stars
        for star in self.bg_stars:
            star_x = (star["pos"][0] - self.world_scroll_x * star["depth"]) % self.W
            star_y = star["pos"][1]
            lum = int(100 * star["depth"])
            pygame.draw.circle(self.screen, (lum, lum, lum), (int(star_x), int(star_y)), 1)
        
        # Draw floor
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (0, self.FLOOR_Y, self.W, self.H - self.FLOOR_Y))

        # Draw finish line
        finish_x = self.LEVEL_LENGTH_PX - self.world_scroll_x
        if finish_x < self.W + 20:
            for i in range(10):
                alpha = 150 - i * 15
                color = (*self.COLOR_FINISH_GLOW, alpha)
                temp_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
                pygame.draw.line(temp_surf, color, (int(finish_x - i), 0), (int(finish_x - i), self.H), 2)
                pygame.draw.line(temp_surf, color, (int(finish_x + i), 0), (int(finish_x + i), self.H), 2)
                self.screen.blit(temp_surf, (0,0))
            pygame.draw.line(self.screen, self.COLOR_FINISH, (int(finish_x), 0), (int(finish_x), self.H), 3)

        # Draw obstacles
        for o in self.obstacles:
            if -o["size"] < o["pos"][0] < self.W + o["size"]:
                x, y, s = int(o["pos"][0]), int(o["pos"][1]), int(o["size"])
                points = [(x, y + s), (x + s, y + s), (x + s/2, y)]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE_GLOW)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)

        # Draw items
        for i in self.bonus_items:
            if -i["size"] < i["pos"][0] < self.W + i["size"]:
                x, y, s = int(i["pos"][0]), int(i["pos"][1]), int(i["size"])
                pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 5
                
                points = []
                for j in range(5):
                    angle = j * (2 * math.pi / 5) * 2 + (self.steps * 0.1)
                    outer_r = s/2 + pulse
                    inner_r = outer_r / 2
                    points.append((x + s/2 + math.cos(angle) * outer_r, y + s/2 + math.sin(angle) * outer_r))
                    angle += (2 * math.pi / 5)
                    points.append((x + s/2 + math.cos(angle) * inner_r, y + s/2 + math.sin(angle) * inner_r))
                
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ITEM_GLOW)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ITEM)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2,2), 2)
            self.screen.blit(temp_surf, (int(p["pos"][0]-2), int(p["pos"][1]-2)))

        # Draw player
        if not self.game_over:
            px, py, ps = int(self.player_pos[0]), int(self.player_pos[1]), int(self.PLAYER_SIZE)
            player_rect = (px, py, ps, ps)
            glow_size = int(ps * 1.5)
            temp_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER_GLOW, 80), (glow_size//2, glow_size//2), glow_size//2)
            self.screen.blit(temp_surf, (px - (glow_size-ps)//2, py - (glow_size-ps)//2))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, player_rect, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.W - score_text.get_width() - 10, 10))

        # Time
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            if self.world_scroll_x >= self.LEVEL_LENGTH_PX:
                msg = "FINISH!"
                color = self.COLOR_FINISH
            else:
                msg = "GAME OVER"
                color = self.COLOR_OBSTACLE
            
            msg_text = self.font_msg.render(msg, True, color)
            text_rect = msg_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(msg_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "world_scroll_x": self.world_scroll_x,
            "progress_percent": (self.world_scroll_x / self.LEVEL_LENGTH_PX) * 100
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # Manual play loop
    pygame.display.set_caption("Neon Runner")
    screen = pygame.display.set_mode((env.W, env.H))
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        env.clock.tick(env.FPS)

    print(f"Game Over. Final Info: {info}")
    env.close()