import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Use a fan to blow marbles into their matching colored slots. Clear obstacles by creating chain reactions."
    user_guide = "Controls: ←→ to aim the fan, ↑↓ to change power. Press space to launch a marble."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen Dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GRID = (30, 40, 55)
        self.COLOR_OBSTACLE = (100, 110, 120)
        self.COLOR_OBSTACLE_OUTLINE = (140, 150, 160)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        self.MARBLE_COLORS = {
            "red": (255, 80, 80),
            "green": (80, 255, 80),
            "blue": (80, 120, 255),
        }
        self.SLOT_COLORS = {
            "red": (180, 50, 50),
            "green": (50, 180, 50),
            "blue": (50, 80, 180),
        }
        
        # Fonts
        try:
            self.font_big = pygame.font.SysFont("monospace", 36, bold=True)
            self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 16)
        except pygame.error:
            self.font_big = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 22)

        # Game Constants
        self.FPS = 60
        self.MAX_EPISODE_STEPS = 60 * self.FPS # 60 seconds
        self.MARBLE_RADIUS = 8
        self.GRAVITY = 0.1
        self.FRICTION = 0.99
        self.BOUNCE_DAMPENING = -0.7
        self.CHAIN_REACTION_THRESHOLD = 10
        self.CHAIN_REACTION_RADIUS = 80
        self.FAN_SPEED_BOOST = 0.1

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.current_level = 0
        self.fan_angle = 0.0
        self.fan_speed = 0.0
        self.fan_pos = (0, 0)
        self.marbles = []
        self.slots = []
        self.obstacles = []
        self.particles = []
        self.marble_queue = []
        self.prev_space_held = False
        
        self.np_random = None

    def _define_levels(self):
        self.levels = {
            1: {
                "slots": [
                    {"pos": (100, 100), "color_name": "red"},
                    {"pos": (self.WIDTH - 100, 100), "color_name": "blue"},
                ],
                "obstacles": [
                    {"rect": pygame.Rect(self.WIDTH // 2 - 20, 150, 40, 100), "vel": (0,0)},
                ]
            },
            2: {
                "slots": [
                    {"pos": (100, 80), "color_name": "red"},
                    {"pos": (self.WIDTH // 2, 80), "color_name": "green"},
                    {"pos": (self.WIDTH - 100, 80), "color_name": "blue"},
                ],
                "obstacles": [
                    {"rect": pygame.Rect(80, 150, 150, 20), "vel": (0,0)},
                    {"rect": pygame.Rect(self.WIDTH - 80 - 150, 150, 150, 20), "vel": (0,0)},
                    {"rect": pygame.Rect(self.WIDTH // 2 - 10, 200, 20, 100), "vel": (0,0)},
                ]
            },
            3: {
                "slots": [
                    {"pos": (100, 80), "color_name": "red"},
                    {"pos": (self.WIDTH // 2, 180), "color_name": "green"},
                    {"pos": (self.WIDTH - 100, 80), "color_name": "blue"},
                ],
                "obstacles": [
                    {"rect": pygame.Rect(50, 150, 100, 20), "vel": (1, 0)},
                    {"rect": pygame.Rect(self.WIDTH - 150, 250, 100, 20), "vel": (-1, 0)},
                    {"rect": pygame.Rect(self.WIDTH // 2 - 60, 50, 120, 20), "vel": (0,0)},
                    {"rect": pygame.Rect(self.WIDTH // 2 - 10, 250, 20, 80), "vel": (0,0)},
                ]
            }
        }

    def _load_level(self, level_num):
        if level_num not in self.levels:
            return False # No more levels

        self.current_level = level_num
        level_data = self.levels[level_num]

        self.slots = []
        for s in level_data["slots"]:
            self.slots.append({
                "pos": s["pos"],
                "color_name": s["color_name"],
                "color": self.SLOT_COLORS[s["color_name"]],
                "rect": pygame.Rect(s["pos"][0] - 25, s["pos"][1] - 25, 50, 50),
                "count": 0
            })

        self.obstacles = []
        for o in level_data["obstacles"]:
            self.obstacles.append({
                "rect": o["rect"].copy(),
                "vel": o["vel"]
            })
        
        self.marbles = []
        self.particles = []
        self._generate_marble_queue()
        return True

    def _generate_marble_queue(self):
        self.marble_queue = []
        num_each_color = 30
        colors = list(self.MARBLE_COLORS.keys())
        for _ in range(num_each_color):
            self.marble_queue.extend(colors)
        random.shuffle(self.marble_queue)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_EPISODE_STEPS
        
        self.fan_pos = (self.WIDTH // 2, self.HEIGHT - 20)
        self.fan_angle = -90.0
        self.fan_speed = 0.5 # 50% power
        
        self.prev_space_held = False

        self._define_levels()
        self._load_level(1)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.time_left -= 1
        self.steps += 1

        # 1. Handle Actions
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: # Up: Increase fan speed
            self.fan_speed = min(2.0, self.fan_speed + 0.02)
        elif movement == 2: # Down: Decrease fan speed
            self.fan_speed = max(0.1, self.fan_speed - 0.02)
        elif movement == 3: # Left: Rotate CCW
            self.fan_angle -= 2.0
        elif movement == 4: # Right: Rotate CW
            self.fan_angle += 2.0
        
        # Launch marble on space press (rising edge)
        if space_held and not self.prev_space_held:
            reward += self._launch_marble()
        self.prev_space_held = space_held

        # 2. Update Game State
        self._update_obstacles()
        marble_updates = self._update_marbles()
        reward += marble_updates["reward"]

        chain_reaction_updates = self._check_slots()
        reward += chain_reaction_updates["reward"]

        if not self.obstacles and self.current_level <= len(self.levels):
            reward += 10 # Level clear bonus
            if not self._load_level(self.current_level + 1):
                self.game_over = True
                reward += 100 # Game win bonus
        
        self._update_particles()
        
        # 3. Check Termination
        terminated = False
        if self.time_left <= 0:
            self.game_over = True
            terminated = True
            if self.current_level <= len(self.levels):
                reward -= 100 # Game loss penalty
        
        if self.game_over:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _launch_marble(self):
        if not self.marble_queue:
            return 0
        
        color_name = self.marble_queue.pop(0)
        angle_rad = math.radians(self.fan_angle)
        power = 5 + self.fan_speed * 10
        
        vel = [math.cos(angle_rad) * power, math.sin(angle_rad) * power]
        pos = [
            self.fan_pos[0] + math.cos(angle_rad) * 30,
            self.fan_pos[1] + math.sin(angle_rad) * 30
        ]
        
        self.marbles.append({
            "pos": pos,
            "vel": vel,
            "color_name": color_name,
            "color": self.MARBLE_COLORS[color_name],
            "radius": self.MARBLE_RADIUS,
            "life": self.FPS * 10 # 10 seconds
        })
        return 0.1 # Reward for launching

    def _update_obstacles(self):
        for o in self.obstacles:
            if o["vel"] != (0,0):
                o["rect"].x += o["vel"][0]
                o["rect"].y += o["vel"][1]
                if o["rect"].left < 0 or o["rect"].right > self.WIDTH:
                    o["vel"] = (-o["vel"][0], o["vel"][1])
                if o["rect"].top < 0 or o["rect"].bottom > self.HEIGHT:
                    o["vel"] = (o["vel"][0], -o["vel"][1])

    def _update_marbles(self):
        reward = 0
        for m in self.marbles[:]:
            m["life"] -= 1
            if m["life"] <= 0:
                self.marbles.remove(m)
                continue

            m["vel"][1] += self.GRAVITY
            m["vel"][0] *= self.FRICTION
            m["vel"][1] *= self.FRICTION
            m["pos"][0] += m["vel"][0]
            m["pos"][1] += m["vel"][1]

            # Wall collisions
            if m["pos"][0] - m["radius"] < 0:
                m["pos"][0] = m["radius"]
                m["vel"][0] *= self.BOUNCE_DAMPENING
            elif m["pos"][0] + m["radius"] > self.WIDTH:
                m["pos"][0] = self.WIDTH - m["radius"]
                m["vel"][0] *= self.BOUNCE_DAMPENING
            if m["pos"][1] - m["radius"] < 0:
                m["pos"][1] = m["radius"]
                m["vel"][1] *= self.BOUNCE_DAMPENING
            elif m["pos"][1] + m["radius"] > self.HEIGHT:
                m["pos"][1] = self.HEIGHT - m["radius"]
                m["vel"][1] *= self.BOUNCE_DAMPENING

            # Obstacle collisions
            for o in self.obstacles:
                if o["rect"].collidepoint(m["pos"]):
                    reward += 0.5
                    # Simple push-out and bounce logic
                    dx = m["pos"][0] - o["rect"].centerx
                    dy = m["pos"][1] - o["rect"].centery
                    if abs(dx) > abs(dy): # Horizontal collision
                        m["vel"][0] *= self.BOUNCE_DAMPENING
                        m["pos"][0] = o["rect"].right + m["radius"] if dx > 0 else o["rect"].left - m["radius"]
                    else: # Vertical collision
                        m["vel"][1] *= self.BOUNCE_DAMPENING
                        m["pos"][1] = o["rect"].bottom + m["radius"] if dy > 0 else o["rect"].top - m["radius"]
        return {"reward": reward}

    def _check_slots(self):
        reward = 0
        for m in self.marbles[:]:
            for s in self.slots:
                if s["rect"].collidepoint(m["pos"]) and s["color_name"] == m["color_name"]:
                    self.marbles.remove(m)
                    s["count"] += 1
                    self.score += 10
                    reward += 1
                    if s["count"] >= self.CHAIN_REACTION_THRESHOLD:
                        reward += self._trigger_chain_reaction(s)
                        s["count"] = 0
                    break
        return {"reward": reward}

    def _trigger_chain_reaction(self, slot):
        reward = 5
        slot_pos = pygame.Vector2(slot["pos"])
        
        # Create particles
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                "pos": list(slot_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "size": random.uniform(2, 5),
                "life": random.randint(20, 40),
                "color": slot["color"]
            })

        # Remove nearby obstacles
        for o in self.obstacles[:]:
            obstacle_pos = pygame.Vector2(o["rect"].center)
            if slot_pos.distance_to(obstacle_pos) < self.CHAIN_REACTION_RADIUS:
                self.obstacles.remove(o)
                self.score += 50
        
        # Boost fan speed
        self.fan_speed = min(2.0, self.fan_speed + self.FAN_SPEED_BOOST)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["size"] *= 0.95

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
            "time_left": self.time_left,
            "current_level": self.current_level
        }

    def _render_text(self, text, font, color, pos, shadow=True, center=False):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect_shadow = text_surf_shadow.get_rect()
            if center: text_rect_shadow.center = (pos[0] + 2, pos[1] + 2)
            else: text_rect_shadow.topleft = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(text_surf_shadow, text_rect_shadow)

        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center: text_rect.center = pos
        else: text_rect.topleft = pos
        self.screen.blit(text_surf, text_rect)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw slots
        for s in self.slots:
            pygame.draw.rect(self.screen, s["color"], s["rect"], 4, border_radius=5)
            fill_ratio = s["count"] / self.CHAIN_REACTION_THRESHOLD
            if fill_ratio > 0:
                fill_h = s["rect"].height * fill_ratio
                fill_rect = pygame.Rect(s["rect"].x, s["rect"].bottom - fill_h, s["rect"].width, fill_h)
                fill_color = list(s["color"])
                fill_color[0] = min(255, fill_color[0] + 50)
                fill_color[1] = min(255, fill_color[1] + 50)
                fill_color[2] = min(255, fill_color[2] + 50)
                pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=5)

        # Draw obstacles
        for o in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, o["rect"], border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, o["rect"], 2, border_radius=3)

        # Draw marbles
        for m in self.marbles:
            pos_int = (int(m["pos"][0]), int(m["pos"][1]))
            # Glow effect
            glow_radius = int(m["radius"] * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, m["color"] + (50,), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))
            # Marble
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], m["radius"], m["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], m["radius"], m["color"])

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 40.0))
            color = p["color"] + (int(alpha),)
            size = int(p["size"])
            if size > 0:
                 # Create a temporary surface for the particle to handle alpha properly
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(particle_surf, size, size, size, color)
                self.screen.blit(particle_surf, (int(p["pos"][0]) - size, int(p["pos"][1]) - size))


        # Draw fan
        fan_color = (255, 255, 0) if self.fan_speed > 1.5 else pygame.Color("cyan").lerp(pygame.Color("yellow"), (self.fan_speed - 0.1) / 1.4)
        angle_rad = math.radians(self.fan_angle)
        p1 = self.fan_pos
        p2 = (p1[0] - 20, p1[1] + 15)
        p3 = (p1[0] + 20, p1[1] + 15)
        pygame.draw.polygon(self.screen, (50, 60, 75), [p1, p2, p3])
        
        indicator_len = 40
        end_pos = (
            self.fan_pos[0] + math.cos(angle_rad) * indicator_len,
            self.fan_pos[1] + math.sin(angle_rad) * indicator_len
        )
        pygame.draw.line(self.screen, fan_color, self.fan_pos, end_pos, 4)

    def _render_ui(self):
        # Level
        self._render_text(f"Level: {self.current_level}", self.font_medium, self.COLOR_TEXT, (10, 5))
        
        # Timer
        secs = self.time_left // self.FPS
        self._render_text(f"Time: {secs}", self.font_medium, self.COLOR_TEXT, (self.WIDTH - 150, 5))

        # Score
        self._render_text(f"Score: {self.score}", self.font_medium, self.COLOR_TEXT, (self.WIDTH // 2, self.HEIGHT - 30), center=True)
        
        # Marble preview
        if self.marble_queue:
            next_color_name = self.marble_queue[0]
            next_color = self.MARBLE_COLORS[next_color_name]
            preview_pos = (40, self.HEIGHT - 40)
            pygame.gfxdraw.aacircle(self.screen, preview_pos[0], preview_pos[1], self.MARBLE_RADIUS, next_color)
            pygame.gfxdraw.filled_circle(self.screen, preview_pos[0], preview_pos[1], self.MARBLE_RADIUS, next_color)
            self._render_text("Next:", self.font_small, self.COLOR_TEXT, (10, self.HEIGHT - 70))

        # Fan speed indicator
        speed_bar_rect = pygame.Rect(self.WIDTH - 150, self.HEIGHT - 25, 140, 15)
        pygame.draw.rect(self.screen, self.COLOR_GRID, speed_bar_rect, border_radius=3)
        speed_fill_width = speed_bar_rect.width * (self.fan_speed / 2.0)
        speed_fill_rect = pygame.Rect(speed_bar_rect.x, speed_bar_rect.y, speed_fill_width, speed_bar_rect.height)
        speed_color = pygame.Color("cyan").lerp(pygame.Color("red"), (self.fan_speed - 0.1) / 1.9)
        pygame.draw.rect(self.screen, speed_color, speed_fill_rect, border_radius=3)
        self._render_text("Power", self.font_small, self.COLOR_TEXT, (self.WIDTH - 150, self.HEIGHT - 45))


        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.current_level > len(self.levels):
                self._render_text("YOU WIN!", self.font_big, self.COLOR_WIN, (self.WIDTH // 2, self.HEIGHT // 2), center=True)
            else:
                self._render_text("TIME'S UP!", self.font_big, self.COLOR_LOSE, (self.WIDTH // 2, self.HEIGHT // 2), center=True)

    def close(self):
        pygame.quit()
    
if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Pygame window for human play
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Marble Blower")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    # Action mapping for human keyboard
    # 0=none, 1=up, 2=down, 3=left, 4=right
    # [movement, space, shift]
    action = [0, 0, 0] 
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get key presses
        keys = pygame.key.get_pressed()
        
        action = [0, 0, 0]
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
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()