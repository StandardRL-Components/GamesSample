import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:01:31.545935
# Source Brief: brief_00159.md
# Brief Index: 159
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls five balloons.
    The goal is to keep all balloons above a certain altitude for a duration
    by managing their vertical movement and switching them between helium/air states.
    Falling stars can be caught for an altitude boost.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Keep five balloons afloat by managing their lift and catching falling stars for a boost. "
        "Maintain all balloons above the win line to achieve stability and win."
    )
    user_guide = (
        "Controls: Use ↑↓ to apply force to the selected balloon. Use ←→ to switch between balloons. "
        "Press space to increase lift (more helium) and shift to decrease lift (more air)."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_BALLOONS = 5
    MAX_STEPS = 1500

    # Game world properties
    ALTITUDE_MAX = 50.0
    ALTITUDE_WIN = 35.0
    WIN_DURATION_STEPS = 300
    SCALE_Y = SCREEN_HEIGHT / ALTITUDE_MAX

    # Balloon physics
    GRAVITY = 0.05
    PLAYER_FORCE = 0.6
    # Buoyancy for each type: 0=Full Helium, 1=Partial Helium, 2=Partial Air, 3=Full Air
    BUOYANCY_LEVELS = [0.20, 0.08, -0.1, -0.25]
    BALLOON_RADIUS = 15
    BALLOON_START_ALT = 20.0

    # Star properties
    STAR_RADIUS = 7
    STAR_SPEED = 1.5
    STAR_BOOST = 5.0
    STAR_SPAWN_BASE_INTERVAL = 40

    # Colors
    COLOR_BG_TOP = (15, 25, 80)
    COLOR_BG_BOTTOM = (60, 100, 180)
    COLOR_ALT_LINE = (255, 255, 255, 50)
    COLOR_SELECTOR = (255, 255, 0, 100)
    COLOR_STAR = (255, 255, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    HELIUM_COLORS = [(80, 150, 255), (100, 170, 255), (120, 190, 255), (140, 210, 255)]
    AIR_COLORS = [(180, 180, 180), (160, 160, 160), (140, 140, 140), (120, 120, 120)]
    BALLOON_TYPE_COLORS = [HELIUM_COLORS[0], HELIUM_COLORS[2], AIR_COLORS[1], AIR_COLORS[3]]

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
        self.font_main = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 16, bold=True)
        
        self.balloons = []
        self.stars = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_balloon_idx = 0
        self.win_timer = 0
        self.star_spawn_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_timer = 0
        self.star_spawn_timer = 0
        self.selected_balloon_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.balloons = []
        for i in range(self.NUM_BALLOONS):
            self.balloons.append({
                "altitude": self.BALLOON_START_ALT + self.np_random.uniform(-5, 5),
                "type": self.np_random.integers(0, len(self.BUOYANCY_LEVELS)), # 0-3
                "x_pos": (self.SCREEN_WIDTH / (self.NUM_BALLOONS + 1)) * (i + 1)
            })

        self.stars = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # 1. Handle Input
        self._handle_input(movement, space_held, shift_held)

        # 2. Update Game Logic
        self._update_balloons(movement)
        self._update_celestial_bodies() # Stars and particles

        # 3. Handle Collisions
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # 4. Calculate Rewards
        win_check_reward = 0
        all_balloons_safe = True
        for balloon in self.balloons:
            if balloon["altitude"] >= self.ALTITUDE_WIN:
                win_check_reward += 0.1
            else:
                all_balloons_safe = False
        reward += win_check_reward
        
        if all_balloons_safe:
            self.win_timer += 1
        else:
            self.win_timer = 0

        # 5. Check Termination Conditions
        terminated = False
        if any(b["altitude"] <= 0 for b in self.balloons):
            reward -= 100  # Loss penalty
            terminated = True
            # sfx: game_loss_sound
        elif self.win_timer >= self.WIN_DURATION_STEPS:
            reward += 100  # Win bonus
            terminated = True
            # sfx: game_win_sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.steps += 1
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Action: Change selected balloon
        if movement == 3: # Left
            self.selected_balloon_idx = (self.selected_balloon_idx - 1 + self.NUM_BALLOONS) % self.NUM_BALLOONS
            # sfx: select_switch
        elif movement == 4: # Right
            self.selected_balloon_idx = (self.selected_balloon_idx + 1) % self.NUM_BALLOONS
            # sfx: select_switch

        # Action: Change balloon type (on press, not hold)
        selected_balloon = self.balloons[self.selected_balloon_idx]
        if space_held and not self.prev_space_held:
            selected_balloon["type"] = max(0, selected_balloon["type"] - 1)
            # sfx: type_change_up
        if shift_held and not self.prev_shift_held:
            selected_balloon["type"] = min(len(self.BUOYANCY_LEVELS) - 1, selected_balloon["type"] + 1)
            # sfx: type_change_down
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_balloons(self, movement):
        # Action: Apply force to selected balloon
        player_force = 0
        if movement == 1: # Up
            player_force = self.PLAYER_FORCE
            # sfx: whoosh_up
        elif movement == 2: # Down
            player_force = -self.PLAYER_FORCE
            # sfx: whoosh_down

        for i, balloon in enumerate(self.balloons):
            buoyancy = self.BUOYANCY_LEVELS[balloon["type"]]
            force = buoyancy - self.GRAVITY
            if i == self.selected_balloon_idx:
                force += player_force
            
            balloon["altitude"] += force
            balloon["altitude"] = max(0, min(self.ALTITUDE_MAX, balloon["altitude"]))

    def _update_celestial_bodies(self):
        # Update stars
        for star in self.stars:
            star["y"] += self.STAR_SPEED
        self.stars = [star for star in self.stars if star["y"] < self.SCREEN_HEIGHT + self.STAR_RADIUS]

        # Spawn new stars
        self.star_spawn_timer += 1
        spawn_interval = max(10, self.STAR_SPAWN_BASE_INTERVAL - self.steps // 50)
        if self.star_spawn_timer >= spawn_interval:
            self.star_spawn_timer = 0
            self.stars.append({
                "x": self.np_random.uniform(self.STAR_RADIUS, self.SCREEN_WIDTH - self.STAR_RADIUS),
                "y": -self.STAR_RADIUS
            })
            
        # Update particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _handle_collisions(self):
        collision_reward = 0
        for balloon in self.balloons:
            balloon_y = self.SCREEN_HEIGHT - (balloon["altitude"] * self.SCALE_Y)
            balloon_pos = pygame.Vector2(balloon["x_pos"], balloon_y)
            
            for star in self.stars[:]:
                star_pos = pygame.Vector2(star["x"], star["y"])
                if balloon_pos.distance_to(star_pos) < self.BALLOON_RADIUS + self.STAR_RADIUS:
                    balloon["altitude"] = min(self.ALTITUDE_MAX, balloon["altitude"] + self.STAR_BOOST)
                    collision_reward += 1.0
                    self.stars.remove(star)
                    self._create_particles(balloon_pos)
                    # sfx: star_collect
                    break # A balloon can only collect one star per frame
        return collision_reward
    
    def _create_particles(self, pos, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan})

    def _get_observation(self):
        # 1. Render Background Gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = [
                self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # 2. Render Altitude Lines
        for alt in range(5, int(self.ALTITUDE_MAX), 5):
            y = self.SCREEN_HEIGHT - (alt * self.SCALE_Y)
            pygame.draw.line(self.screen, self.COLOR_ALT_LINE, (0, y), (self.SCREEN_WIDTH, y), 1)

        # 3. Render Stars
        for star in self.stars:
            pos = (int(star["x"]), int(star["y"]))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.STAR_RADIUS + 3, (*self.COLOR_STAR, 50))
            # Star body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.STAR_RADIUS, self.COLOR_STAR)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.STAR_RADIUS, self.COLOR_STAR)

        # 4. Render Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (*self.COLOR_STAR, alpha)
            try: # Use a try-except block for Surface.set_at as it can be picky with alpha
                surf = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (2,2), 2)
                self.screen.blit(surf, (int(p['pos'][0])-2, int(p['pos'][1])-2), special_flags=pygame.BLEND_RGBA_ADD)
            except: # Fallback to non-alpha circle
                pygame.draw.circle(self.screen, self.COLOR_STAR, (int(p['pos'][0]), int(p['pos'][1])), 2)


        # 5. Render Balloons
        for i, balloon in enumerate(self.balloons):
            y_pos = self.SCREEN_HEIGHT - (balloon["altitude"] * self.SCALE_Y)
            x_pos = balloon["x_pos"]
            pos = (int(x_pos), int(y_pos))
            color = self.BALLOON_TYPE_COLORS[balloon["type"]]

            # Selector glow
            if i == self.selected_balloon_idx:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALLOON_RADIUS + 5, self.COLOR_SELECTOR)

            # String
            string_end_y = min(self.SCREEN_HEIGHT, y_pos + self.BALLOON_RADIUS + 20)
            pygame.draw.aaline(self.screen, (255, 255, 255), (x_pos, y_pos + self.BALLOON_RADIUS), (x_pos, string_end_y))
            
            # Balloon body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALLOON_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALLOON_RADIUS, (0,0,0,50))
            
            # Highlight
            highlight_pos = (pos[0] - self.BALLOON_RADIUS // 3, pos[1] - self.BALLOON_RADIUS // 3)
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], self.BALLOON_RADIUS // 3, (255,255,255,50))
            
            # Altitude text
            alt_text = f"{balloon['altitude']:.1f}"
            self._render_text(alt_text, (pos[0], pos[1] - self.BALLOON_RADIUS - 15), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, font=self.font_small, center=True)

        # 6. Render UI
        score_text = f"SCORE: {self.score:.1f}"
        self._render_text(score_text, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        win_progress = self.win_timer / self.WIN_DURATION_STEPS
        win_text = f"STABILITY: {int(win_progress * 100)}%"
        self._render_text(win_text, (self.SCREEN_WIDTH - 10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="right")

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_text(self, text, pos, color, shadow_color, font=None, center=False, align="left"):
        if font is None:
            font = self.font_main
            
        shadow_surf = font.render(text, True, shadow_color)
        text_surf = font.render(text, True, color)
        
        if center:
             text_rect = text_surf.get_rect(center=pos)
        elif align == "right":
            text_rect = text_surf.get_rect(topright=pos)
        else: # align == "left"
            text_rect = text_surf.get_rect(topleft=pos)
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win_timer": self.win_timer,
            "balloon_altitudes": [b['altitude'] for b in self.balloons],
            "balloon_types": [b['type'] for b in self.balloons],
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will not be executed by the evaluation script.
    # Ensure `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` is commented out for local play.
    # os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Balloon Juggler")
    screen_display = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0.0
    
    # Remove the validation call from the main loop
    # env.validate_implementation() 
    
    while not terminated:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("--- Game Reset ---")
                elif event.key == pygame.K_ESCAPE:
                    terminated = True


        env.clock.tick(30) # Limit frame rate for human playability

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()