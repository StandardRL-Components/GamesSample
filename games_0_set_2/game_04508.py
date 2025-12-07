import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade-style Gymnasium environment where the player pilots a ship through a
    dangerous asteroid field. The goal is to survive for 60 seconds while dodging
    asteroids. The game's difficulty increases over time with more and faster asteroids.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to move your ship. "
        "Survive for 60 seconds to win."
    )

    game_description = (
        "Pilot a ship through an asteroid field, dodging incoming rocks for 60 seconds to survive."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_THRUSTER = (255, 180, 50)
        self.COLOR_ASTEROID = (150, 150, 150)
        self.COLOR_UI = (0, 255, 150)
        self.COLOR_WIN = (50, 255, 50)
        self.COLOR_LOSE = (255, 50, 50)

        # --- Game Parameters ---
        self.fps = 60
        self.max_duration_seconds = 60
        self.max_steps = self.max_duration_seconds * self.fps
        
        self.player_speed = 5.0
        self.player_radius = 12
        
        self.initial_spawn_rate = 5.0  # Asteroids per second
        self.spawn_rate_increase = 0.01 # Per second
        
        self.initial_asteroid_speed = 2.0
        self.asteroid_speed_increase_per_10s = 0.2
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self.player_pos = None
        self.asteroids = []
        self.particles = []
        self.starfield = []
        self.screen_flash = 0
        
        self._create_starfield()
        # self.reset() is called by the validation function
        
        # self.validate_implementation() # This will be called by the verifier, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False

        self.player_pos = pygame.math.Vector2(self.width / 2, self.height - 50)
        
        self.asteroids.clear()
        self.particles.clear()
        self.screen_flash = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        # --- Action Handling ---
        movement = action[0]
        
        if self.auto_advance:
            self.clock.tick(self.fps)
            
        reward = 0.0
        terminated = False
        
        if not self.game_over:
            # --- Update Game Logic ---
            self._update_player(movement)
            self._update_asteroids()
            self._update_particles()
            
            # --- Collision & Termination Check ---
            collided = self._check_collisions()
            time_up = self._check_time_limit()
            terminated = collided or time_up
            
            # --- Reward Calculation ---
            if terminated and self.win:
                reward = 100.0 # Win bonus
                self.game_over = True
            elif terminated and not self.win:
                reward = -100.0 # Collision penalty
                self.game_over = True
            else:
                reward = 0.1 # Survival reward per step
            
            self.score += reward

        else: # Game is over, just wait
            self._update_particles()
            terminated = True
            
        # --- Return state ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always False in this environment
            self._get_info()
        )

    def _update_player(self, movement):
        moved = False
        if movement == 1: # Up
            self.player_pos.y -= self.player_speed
            moved = True
        elif movement == 2: # Down
            self.player_pos.y += self.player_speed
            moved = True
        elif movement == 3: # Left
            self.player_pos.x -= self.player_speed
            moved = True
        elif movement == 4: # Right
            self.player_pos.x += self.player_speed
            moved = True
            
        # Clamp to screen
        self.player_pos.x = np.clip(self.player_pos.x, self.player_radius, self.width - self.player_radius)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_radius, self.height - self.player_radius)

        # Spawn thruster particles
        if moved:
            # sfx: player_thruster_loop
            for _ in range(2):
                vel_angle = self.np_random.uniform(-0.2, 0.2)
                vel_mag = self.np_random.uniform(2, 4)
                
                if movement == 1: v = pygame.math.Vector2(0, vel_mag).rotate_rad(vel_angle)
                elif movement == 2: v = pygame.math.Vector2(0, -vel_mag).rotate_rad(vel_angle)
                elif movement == 3: v = pygame.math.Vector2(vel_mag, 0).rotate_rad(vel_angle)
                else: v = pygame.math.Vector2(-vel_mag, 0).rotate_rad(vel_angle)

                self.particles.append({
                    "pos": pygame.math.Vector2(self.player_pos),
                    "vel": v,
                    "radius": self.np_random.uniform(2, 5),
                    "color": self.COLOR_THRUSTER,
                    "lifetime": 15
                })

    def _update_asteroids(self):
        # --- Spawn new asteroids ---
        seconds_passed = self.steps / self.fps
        current_spawn_rate = self.initial_spawn_rate + self.spawn_rate_increase * seconds_passed
        
        if self.np_random.random() < current_spawn_rate / self.fps:
            radius = self.np_random.integers(10, 35)
            pos = pygame.math.Vector2(
                self.np_random.uniform(radius, self.width - radius),
                -radius
            )
            
            speed_multiplier = 1.0 + (self.asteroid_speed_increase_per_10s * (self.steps // (self.fps * 10)))
            speed = self.initial_asteroid_speed * speed_multiplier
            
            angle = self.np_random.uniform(-0.3, 0.3) # Slight side-to-side motion
            vel = pygame.math.Vector2(0, speed).rotate_rad(angle)
            
            color_variation = self.np_random.integers(-20, 20)
            color = tuple(np.clip([c + color_variation for c in self.COLOR_ASTEROID], 80, 180))
            
            self.asteroids.append({"pos": pos, "vel": vel, "radius": radius, "color": color})

        # --- Move and despawn existing asteroids ---
        for asteroid in self.asteroids[:]:
            asteroid["pos"] += asteroid["vel"]
            if asteroid["pos"].y > self.height + asteroid["radius"]:
                self.asteroids.remove(asteroid)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            p["radius"] *= 0.95
            if p["lifetime"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    def _check_collisions(self):
        for asteroid in self.asteroids:
            if self.player_pos is None: continue
            dist = self.player_pos.distance_to(asteroid["pos"])
            if dist < self.player_radius + asteroid["radius"]:
                self._create_explosion()
                self.game_over = True
                self.win = False
                return True
        return False

    def _check_time_limit(self):
        self.steps += 1
        if self.steps >= self.max_steps:
            self.game_over = True
            self.win = True
            return True
        return False
        
    def _create_explosion(self):
        # sfx: player_explosion
        self.screen_flash = 20 # Flash for 20 frames
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 8)
            vel = pygame.math.Vector2(speed, 0).rotate_rad(angle)
            self.particles.append({
                "pos": pygame.math.Vector2(self.player_pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 6),
                "color": random.choice([self.COLOR_LOSE, self.COLOR_THRUSTER, self.COLOR_PLAYER]),
                "lifetime": self.np_random.integers(30, 60)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_starfield()
        self._render_asteroids()
        if not (self.game_over and not self.win): # Don't draw player if they exploded
            self._render_player()
        self._render_particles()
        self._render_ui()
        
        if self.screen_flash > 0:
            flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            alpha = int(255 * (self.screen_flash / 20))
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.screen_flash -= 1

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self):
        # Use np_random if available, otherwise fallback to random for initial call
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else random
        for star in self.starfield:
            star['y'] += star['speed']
            if star['y'] > self.height:
                star['y'] = 0
                star['x'] = rng.uniform(0, self.width)
            
            color_val = int(star['brightness'] * (1 - (star['speed'] / 3.0)) * 200)
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, (int(star['x']), int(star['y'])), int(star['size']))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            x, y = int(asteroid["pos"].x), int(asteroid["pos"].y)
            r = int(asteroid["radius"])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, asteroid["color"])
            pygame.gfxdraw.aacircle(self.screen, x, y, r, asteroid["color"])
            
            # Add a simple crater for detail
            crater_r = max(2, int(r * 0.3))
            crater_offset_x = int(r * 0.4)
            crater_offset_y = -int(r * 0.4)
            crater_color = tuple(np.clip([c-40 for c in asteroid["color"]], 0, 255))
            pygame.gfxdraw.filled_circle(self.screen, x + crater_offset_x, y + crater_offset_y, crater_r, crater_color)
            pygame.gfxdraw.aacircle(self.screen, x + crater_offset_x, y + crater_offset_y, crater_r, crater_color)


    def _render_player(self):
        if self.player_pos is None: return
        p = self.player_pos
        r = self.player_radius
        points = [
            (p.x, p.y - r),
            (p.x - r * 0.8, p.y + r * 0.8),
            (p.x + r * 0.8, p.y + r * 0.8)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
    
    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), p["color"])

    def _render_ui(self):
        # Score display
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Timer display
        time_left = max(0, (self.max_steps - self.steps) / self.fps)
        timer_text = f"TIME: {time_left:.2f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(timer_surf, (self.width - timer_surf.get_width() - 10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            if self.win:
                msg_text = "SURVIVED!"
                msg_color = self.COLOR_WIN
            else:
                msg_text = "GAME OVER"
                msg_color = self.COLOR_LOSE
            
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.max_steps - self.steps) / self.fps),
            "asteroids_on_screen": len(self.asteroids)
        }
        
    def _create_starfield(self):
        self.starfield.clear()
        for _ in range(150): # Number of stars
            self.starfield.append({
                'x': random.uniform(0, self.width),
                'y': random.uniform(0, self.height),
                'speed': random.uniform(0.5, 2.5), # Parallax effect
                'size': random.uniform(0.5, 1.5),
                'brightness': random.uniform(0.5, 1.0)
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    # Note: This will run headlessly. To see the game, you need to
    # remove the os.environ line at the top and run on a system with a display.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        print("Running in headless mode. No visual output will be shown.")
        print("To play visually, comment out `os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")`")
        # A simple loop to test the environment without visuals
        env = GameEnv()
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished. Info: {info}")
                obs, info = env.reset()
        env.close()
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # --- Pygame setup for human play ---
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Asteroid Survival")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- Action Mapping for Human ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Rendering ---
            # The observation is already a rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward}")
                print("Resetting in 3 seconds...")
                pygame.time.wait(3000)
                obs, info = env.reset()
                total_reward = 0
                
            clock.tick(env.fps)
            
        env.close()