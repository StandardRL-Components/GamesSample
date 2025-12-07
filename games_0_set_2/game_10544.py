import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:33:26.619887
# Source Brief: brief_00544.md
# Brief Index: 544
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must restore rhythm to a grotesque
    clockwork mechanism by launching cogs into the correct sockets.

    **Visual Style:** Grotesque clockwork, metallic shades, unsettling organic textures.
    **Core Gameplay:** Aim a reticle, launch a cog, and observe the effect on the
    clockwork's rhythm, visualized by a pulsating heart.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Launch cogs into the correct sockets to restore the rhythm of a grotesque clockwork heart."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim the reticle. Press space to launch a cog. Hold shift to reset the reticle's position."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.W, self.H = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 28, bold=True)

        # --- Colors & Visuals ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_BG_GEAR = (30, 35, 40)
        self.COLOR_RETICLE = (255, 255, 0)
        self.COLOR_TRAJECTORY = (255, 255, 0, 100)
        self.COLOR_COG = (200, 200, 210)
        self.COLOR_COG_TEETH = (160, 160, 170)
        self.COLOR_SOCKET_CORRECT = (40, 100, 40)
        self.COLOR_SOCKET_INCORRECT = (100, 40, 40)
        self.COLOR_SOCKET_FILLED_CORRECT = (80, 220, 80)
        self.COLOR_SOCKET_FILLED_INCORRECT = (220, 80, 80)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_HEART_STABLE = (80, 220, 80)
        self.COLOR_HEART_UNSTABLE = (220, 80, 80)
        self.PARTICLE_COLOR_GOOD = (255, 255, 100)
        self.PARTICLE_COLOR_BAD = (150, 150, 150)

        # --- Game State Variables ---
        self.steps = None
        self.score = None
        self.level = None
        self.game_over = None
        self.rhythm = None
        self.cogs_to_launch = None
        self.prev_space_held = None
        self.reticle_pos = None
        self.cog_launch_pos = None
        self.sockets = None
        self.launched_cogs = None
        self.particles = None
        self.bg_gears = None

        # --- Game Parameters ---
        self.MAX_STEPS = 5000
        self.RETICLE_SPEED = 8
        self.COG_SPEED = 12
        self.COG_RADIUS = 12
        self.SOCKET_RADIUS = 15

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.rhythm = 0.5  # 0.0 (unstable) to 1.0 (stable)
        self.prev_space_held = False

        self.reticle_pos = pygame.Vector2(self.W / 2, self.H / 2)
        self.cog_launch_pos = pygame.Vector2(self.W / 2, self.H - 30)

        self.launched_cogs = []
        self.particles = []
        
        self._generate_bg_gears()
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        old_rhythm = self.rhythm

        # --- 1. Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- 2. Update Game State ---
        reward += self._update_cogs()
        self._update_particles()
        
        # --- 3. Calculate Reward & Check Termination ---
        # Continuous reward for rhythm change
        rhythm_change = self.rhythm - old_rhythm
        reward += rhythm_change * 20 

        terminated = self._check_termination()
        if terminated:
            if self.rhythm <= 0.01:
                reward = -100  # Failure
            elif self.steps >= self.MAX_STEPS:
                reward = -50 # Timeout
            # Level completion reward is handled in _check_level_complete

        self.prev_space_held = space_held

        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Reticle movement
        if movement == 1: self.reticle_pos.y -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos.y += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos.x -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos.x += self.RETICLE_SPEED
        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.W)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.H)

        # Reticle reset
        if shift_held:
            self.reticle_pos = pygame.Vector2(self.W / 2, self.H / 2)

        # Cog launch
        is_launching = space_held and not self.prev_space_held
        if is_launching and self.cogs_to_launch > 0:
            self.cogs_to_launch -= 1
            direction = (self.reticle_pos - self.cog_launch_pos).normalize()
            self.launched_cogs.append({
                "pos": self.cog_launch_pos.copy(),
                "vel": direction * self.COG_SPEED,
                "rotation": 0,
                "rotation_speed": random.uniform(-5, 5)
            })
            # sfx: cog_launch.wav

    def _update_cogs(self):
        step_reward = 0
        cogs_to_remove = []
        for cog in self.launched_cogs:
            cog["pos"] += cog["vel"]
            cog["rotation"] += cog["rotation_speed"]

            # Check for collision with sockets
            for socket in self.sockets:
                if not socket["is_filled"]:
                    dist = cog["pos"].distance_to(socket["pos"])
                    if dist < self.SOCKET_RADIUS:
                        socket["is_filled"] = True
                        socket["filled_color"] = self.COLOR_COG
                        cogs_to_remove.append(cog)
                        
                        if socket["is_correct"]:
                            # sfx: positive_click.wav
                            self.rhythm += 0.2
                            step_reward += 5
                            self._create_particles(socket["pos"], self.PARTICLE_COLOR_GOOD, 30)
                        else:
                            # sfx: negative_clunk.wav
                            self.rhythm -= 0.3
                            step_reward -= 5
                            self._create_particles(socket["pos"], self.PARTICLE_COLOR_BAD, 20)
                        break # Cog can only fill one socket

            # Remove cogs that go off-screen
            if not (0 < cog["pos"].x < self.W and 0 < cog["pos"].y < self.H):
                cogs_to_remove.append(cog)
        
        self.launched_cogs = [c for c in self.launched_cogs if c not in cogs_to_remove]
        self.rhythm = np.clip(self.rhythm, 0, 1)
        return step_reward

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _check_termination(self):
        if self.rhythm <= 0.01:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        
        # Check for level completion
        correct_sockets = [s for s in self.sockets if s["is_correct"]]
        if all(s["is_filled"] for s in correct_sockets):
            # sfx: level_complete.wav
            self.score += 100 # Use score as a proxy for level completion reward in info
            self.level += 1
            if self.level > 10: # Win condition
                self.game_over = True
                return True
            self._generate_level()
            self.rhythm = 0.5 # Reset rhythm for next level

        return False

    def _generate_level(self):
        self.sockets = []
        self.launched_cogs.clear()
        self.particles.clear()
        
        num_correct = min(self.level, 5) 
        num_incorrect = min(self.level + 1, 6)
        num_sockets = num_correct + num_incorrect
        self.cogs_to_launch = num_correct

        candidate_positions = []
        attempts = 0
        while len(candidate_positions) < num_sockets and attempts < 500:
            attempts += 1
            pos = pygame.Vector2(
                random.uniform(self.W * 0.1, self.W * 0.9),
                random.uniform(self.H * 0.1, self.H * 0.7)
            )
            
            # Ensure sockets are not too close to each other
            too_close = False
            for existing_pos in candidate_positions:
                if pos.distance_to(existing_pos) < self.SOCKET_RADIUS * 3:
                    too_close = True
                    break
            if not too_close:
                candidate_positions.append(pos)
        
        for i, pos in enumerate(candidate_positions):
            is_correct = i < num_correct
            self.sockets.append({
                "pos": pos,
                "is_correct": is_correct,
                "is_filled": False,
                "filled_color": None
            })
        random.shuffle(self.sockets)

    def _generate_bg_gears(self):
        self.bg_gears = []
        for _ in range(10):
            self.bg_gears.append({
                "pos": pygame.Vector2(random.randint(0, self.W), random.randint(0, self.H)),
                "radius": random.randint(20, 80),
                "teeth": random.randint(8, 20),
                "rotation": random.uniform(0, 360)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "rhythm": self.rhythm,
            "cogs_left": self.cogs_to_launch
        }

    def _render_background(self):
        for gear in self.bg_gears:
            self._draw_cog(
                self.screen, 
                self.COLOR_BG_GEAR, 
                self.COLOR_BG_GEAR, 
                gear["pos"], 
                gear["radius"], 
                gear["teeth"], 
                gear["rotation"]
            )
        # Pulsating organic veins effect
        for i in range(4):
            phase = (self.steps * 0.02 + i * math.pi/2)
            instability = 1.0 - self.rhythm
            amp = 5 + 20 * instability
            freq = 0.02
            
            points = []
            for x in range(0, self.W, 20):
                y_offset = self.H * (0.2 * i + 0.1)
                y = y_offset + math.sin(x * freq + phase) * amp
                points.append((x, y))
            
            if len(points) > 1:
                color = self.COLOR_HEART_UNSTABLE if instability > 0.5 else self.COLOR_SOCKET_INCORRECT
                pygame.draw.lines(self.screen, color, False, points, int(1 + 4 * instability))


    def _render_game(self):
        # Trajectory line
        self._draw_dashed_line(self.cog_launch_pos, self.reticle_pos, self.COLOR_TRAJECTORY)

        # Sockets
        for s in self.sockets:
            color = self.COLOR_SOCKET_INCORRECT
            if s["is_correct"]:
                color = self.COLOR_SOCKET_CORRECT
            if s["is_filled"]:
                color = self.COLOR_SOCKET_FILLED_INCORRECT
                if s["is_correct"]:
                    color = self.COLOR_SOCKET_FILLED_CORRECT

            pygame.gfxdraw.aacircle(self.screen, int(s["pos"].x), int(s["pos"].y), self.SOCKET_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, int(s["pos"].x), int(s["pos"].y), self.SOCKET_RADIUS, color)
            if s["is_filled"]:
                 self._draw_cog(self.screen, s["filled_color"], self.COLOR_COG_TEETH, s["pos"], self.COG_RADIUS, 8, 0)


        # Launched Cogs
        for cog in self.launched_cogs:
            self._draw_cog(self.screen, self.COLOR_COG, self.COLOR_COG_TEETH, cog["pos"], self.COG_RADIUS, 8, cog["rotation"])

        # Reticle
        px, py = int(self.reticle_pos.x), int(self.reticle_pos.y)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (px - 10, py), (px + 10, py), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (px, py - 10), (px, py + 10), 2)

        # Particles
        for p in self.particles:
            alpha = max(0, p["lifetime"] * 5)
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)


    def _render_ui(self):
        # Rhythm Heart
        pulse = math.sin(self.steps * (0.1 + 0.4 * (1 - self.rhythm)))
        heart_size = 25 + 5 * pulse
        heart_color = self.COLOR_HEART_STABLE if self.rhythm > 0.5 else self.COLOR_HEART_UNSTABLE
        heart_color = tuple(int(a + (b - a) * (1-self.rhythm)) for a, b in zip(self.COLOR_HEART_STABLE, self.COLOR_HEART_UNSTABLE))
        
        heart_pos_x, heart_pos_y = 50, 40
        p1 = (heart_pos_x, heart_pos_y - heart_size // 4)
        p2 = (heart_pos_x + heart_size // 2, heart_pos_y - heart_size // 2)
        p3 = (heart_pos_x + heart_size, heart_pos_y - heart_size // 4)
        p4 = (heart_pos_x, heart_pos_y + heart_size)
        p5 = (heart_pos_x - heart_size, heart_pos_y - heart_size // 4)
        p6 = (heart_pos_x - heart_size // 2, heart_pos_y - heart_size // 2)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p3, p4, p5], heart_color)
        pygame.gfxdraw.filled_circle(self.screen, int(p2[0]), int(p2[1]), int(heart_size * 0.55), heart_color)
        pygame.gfxdraw.filled_circle(self.screen, int(p6[0]), int(p6[1]), int(heart_size * 0.55), heart_color)

        # Text info
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        cogs_text = self.font_ui.render(f"COGS: {self.cogs_to_launch}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.W - 150, 20))
        self.screen.blit(level_text, (self.W - 150, 45))
        self.screen.blit(cogs_text, (self.W - 150, 70))
        
        if self.game_over:
            msg = "MECHANISM DESTABILIZED" if self.rhythm <= 0.01 else "SESSION TIMEOUT"
            if self.level > 10: msg = "MECHANISM STABILIZED"
            end_text = self.font_title.render(msg, True, self.COLOR_HEART_UNSTABLE if self.rhythm <= 0.01 else self.COLOR_HEART_STABLE)
            text_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, text_rect)


    def _draw_cog(self, surface, color_body, color_teeth, pos, radius, num_teeth, rotation):
        # Draw main body
        px, py = int(pos.x), int(pos.y)
        pygame.gfxdraw.aacircle(surface, px, py, radius, color_body)
        pygame.gfxdraw.filled_circle(surface, px, py, radius, color_body)

        # Draw teeth
        tooth_length = radius * 1.3
        for i in range(num_teeth):
            angle = math.radians(i * (360 / num_teeth) + rotation)
            start_pos = (
                px + radius * math.cos(angle),
                py + radius * math.sin(angle)
            )
            end_pos = (
                px + tooth_length * math.cos(angle),
                py + tooth_length * math.sin(angle)
            )
            pygame.draw.line(surface, color_teeth, start_pos, end_pos, 3)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "lifetime": random.randint(20, 40),
                "radius": random.uniform(1, 4),
                "color": color
            })
            
    def _draw_dashed_line(self, start_pos, end_pos, color, dash_length=10):
        path = end_pos - start_pos
        length = path.length()
        if length == 0: return
        direction = path.normalize()
        
        current_pos = start_pos.copy()
        traveled = 0
        drawing = True
        while traveled < length:
            segment_end = traveled + dash_length
            if segment_end > length:
                segment_end = length
            
            p1 = start_pos + direction * traveled
            p2 = start_pos + direction * segment_end

            if drawing:
                pygame.draw.line(self.screen, color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 1)
            
            traveled = segment_end + dash_length
            drawing = not drawing

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # It will not be used for grading.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # For human play, we want a visible screen
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Cog Rhapsody")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space = 0    # 0=released, 1=held
        shift = 0    # 0=released, 1=held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Reason: {('Terminated' if terminated else 'Truncated')}")
            # Optionally, you can auto-reset here or wait for 'r' key
            # obs, info = env.reset()

        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()