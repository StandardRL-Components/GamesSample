
# Generated: 2025-08-28T01:59:07.674509
# Source Brief: brief_01567.md
# Brief Index: 1567

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space to activate your mining beam on asteroids. "
        "Avoid the red lasers from enemy turrets!"
    )

    game_description = (
        "Pilot a mining ship, collect minerals from asteroids while dodging enemy laser fire "
        "in a top-down arcade environment. Collect 500 minerals to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 2000, 2000

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_THRUSTER = (255, 180, 50)
        self.COLOR_ASTEROID = (120, 120, 120)
        self.COLOR_LASER = (255, 50, 50)
        self.COLOR_MINERAL = (255, 220, 0)
        self.COLOR_TURRET = (200, 0, 100)
        self.COLOR_TURRET_CHARGE = (255, 100, 200)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_BEAM = (100, 255, 100, 150)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # Game constants
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 500
        self.INITIAL_LIVES = 3
        self.PLAYER_ACCELERATION = 0.4
        self.PLAYER_DRAG = 0.95
        self.PLAYER_MAX_SPEED = 6
        self.PLAYER_RADIUS = 12
        self.BEAM_LENGTH = 150

        # Will be initialized in reset()
        self.np_random = None
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.last_move_dir = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.asteroids = None
        self.turrets = None
        self.lasers = None
        self.particles = None
        self.stars = None
        self.laser_fire_period = None
        self.asteroid_spawn_period = None
        self.asteroid_spawn_timer = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_angle = -math.pi / 2
        self.last_move_dir = np.array([0, -1]) # Start facing up

        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False

        self.asteroids = []
        self.turrets = []
        self.lasers = []
        self.particles = []

        self.laser_fire_period = 90  # in steps (3 seconds at 30fps)
        self.asteroid_spawn_period = 150 # in steps (5 seconds at 30fps)
        self.asteroid_spawn_timer = self.asteroid_spawn_period

        self._initialize_world()

        return self._get_observation(), self._get_info()

    def _initialize_world(self):
        # Create a static starfield
        self.stars = []
        for _ in range(200):
            self.stars.append(
                (
                    self.np_random.integers(0, self.WORLD_WIDTH),
                    self.np_random.integers(0, self.WORLD_HEIGHT),
                    self.np_random.choice([1, 2]),
                    self.np_random.integers(100, 200)
                )
            )
        # Place turrets
        turret_positions = [
            (300, 300), (self.WORLD_WIDTH - 300, 300),
            (300, self.WORLD_HEIGHT - 300), (self.WORLD_WIDTH - 300, self.WORLD_HEIGHT - 300),
            (self.WORLD_WIDTH / 2, 200), (self.WORLD_WIDTH / 2, self.WORLD_HEIGHT - 200),
        ]
        for pos in turret_positions:
            self.turrets.append({
                "pos": np.array(pos, dtype=float),
                "fire_timer": self.np_random.integers(0, self.laser_fire_period),
                "charging": False
            })
        # Spawn initial asteroids
        for _ in range(10):
            self._spawn_asteroid()

    def _spawn_asteroid(self, at_pos=None):
        if at_pos is None:
            # Spawn off-screen
            edge = self.np_random.integers(0, 4)
            if edge == 0: # top
                pos = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), -50])
            elif edge == 1: # bottom
                pos = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.WORLD_HEIGHT + 50])
            elif edge == 2: # left
                pos = np.array([-50, self.np_random.uniform(0, self.WORLD_HEIGHT)])
            else: # right
                pos = np.array([self.WORLD_WIDTH + 50, self.np_random.uniform(0, self.WORLD_HEIGHT)])
        else:
            pos = at_pos
        
        size = self.np_random.uniform(20, 40)
        vel = self.np_random.uniform(-0.5, 0.5, size=2)
        
        num_vertices = self.np_random.integers(7, 12)
        vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            radius = size * self.np_random.uniform(0.8, 1.2)
            vertices.append((radius * math.cos(angle), radius * math.sin(angle)))

        self.asteroids.append({
            "pos": pos, "vel": vel, "size": size,
            "minerals": int(size * 2), "max_minerals": int(size * 2),
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": self.np_random.uniform(-0.02, 0.02),
            "vertices": vertices
        })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1

        reward_events = {"minerals": 0, "asteroid_destroyed": 0, "life_lost": 0}

        self._handle_input(movement)
        self._update_physics()
        self._update_turrets()
        self._update_lasers(reward_events)
        self._update_asteroids(space_held, reward_events)
        self._update_particles()
        self._spawn_entities()
        self._update_difficulty()
        
        self.steps += 1
        reward = self._calculate_reward(reward_events, space_held)
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1  # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1  # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.player_vel += move_vec * self.PLAYER_ACCELERATION
            self.last_move_dir = move_vec

    def _update_physics(self):
        # Player physics
        self.player_vel *= self.PLAYER_DRAG
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
        self.player_pos += self.player_vel
        
        # Clamp player to world bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WORLD_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.WORLD_HEIGHT)

        # Update asteroid physics
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"]
            asteroid["angle"] += asteroid["rot_speed"]
            # Bounce off world edges
            if not (0 < asteroid["pos"][0] < self.WORLD_WIDTH): asteroid["vel"][0] *= -1
            if not (0 < asteroid["pos"][1] < self.WORLD_HEIGHT): asteroid["vel"][1] *= -1

    def _update_turrets(self):
        for turret in self.turrets:
            turret["fire_timer"] -= 1
            if turret["fire_timer"] <= -30: # Charge duration (1 sec)
                turret["charging"] = False
                # Fire
                # SFX: LaserFire.wav
                direction = (self.player_pos - turret["pos"])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                self.lasers.append({
                    "pos": turret["pos"].copy(),
                    "dir": direction,
                    "speed": 8,
                    "life": 200 # Lifetime in steps
                })
                turret["fire_timer"] = self.laser_fire_period
            elif turret["fire_timer"] <= 0:
                turret["charging"] = True

    def _update_lasers(self, reward_events):
        for laser in self.lasers[:]:
            laser["pos"] += laser["dir"] * laser["speed"]
            laser["life"] -= 1
            if laser["life"] <= 0:
                self.lasers.remove(laser)
                continue
            
            # Collision with player
            if np.linalg.norm(laser["pos"] - self.player_pos) < self.PLAYER_RADIUS:
                self.lasers.remove(laser)
                self.lives -= 1
                reward_events["life_lost"] += 1
                # SFX: PlayerExplosion.wav
                self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)
                self.player_vel *= 0.1 # Stun player
                break

    def _update_asteroids(self, space_held, reward_events):
        beam_target = None
        if space_held:
            beam_start = self.player_pos
            beam_end = self.player_pos + self.last_move_dir * self.BEAM_LENGTH
            
            min_dist = float('inf')
            for asteroid in self.asteroids:
                # Simple line-circle intersection
                dist_to_line = np.linalg.norm(np.cross(beam_end - beam_start, beam_start - asteroid["pos"])) / np.linalg.norm(beam_end - beam_start)
                if dist_to_line < asteroid["size"]:
                    dist_to_player = np.linalg.norm(asteroid["pos"] - self.player_pos)
                    if dist_to_player < min_dist and dist_to_player < self.BEAM_LENGTH + asteroid["size"]:
                        min_dist = dist_to_player
                        beam_target = asteroid

        if beam_target and beam_target["minerals"] > 0:
            # SFX: Mining.wav (loop)
            beam_target["minerals"] -= 1
            self.score = min(self.WIN_SCORE, self.score + 1)
            reward_events["minerals"] += 1
            if self.np_random.random() < 0.2:
                self._create_mineral_particles(beam_target["pos"], 1)

            if beam_target["minerals"] <= 0:
                # SFX: AsteroidExplosion.wav
                self._create_explosion(beam_target["pos"], int(beam_target["size"]), self.COLOR_ASTEROID)
                self.asteroids.remove(beam_target)
                reward_events["asteroid_destroyed"] += 1
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_entities(self):
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroid()
            self.asteroid_spawn_timer = self.asteroid_spawn_period

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.laser_fire_period = max(30, self.laser_fire_period - 1.5) # 0.05s * 30fps
        if self.steps > 0 and self.steps % 200 == 0:
            self.asteroid_spawn_period = max(60, self.asteroid_spawn_period - 3) # 0.1s * 30fps

    def _calculate_reward(self, events, space_held):
        reward = 0
        reward += events["minerals"] * 0.1
        reward += events["asteroid_destroyed"] * 1.0
        reward += events["life_lost"] * -10.0
        if not space_held and events["minerals"] == 0:
            reward -= 0.01
        return reward

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

    def _world_to_screen(self, pos):
        screen_pos = pos - self.player_pos + np.array([self.WIDTH / 2, self.HEIGHT / 2])
        return screen_pos.astype(int)

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": self.np_random.integers(15, 30),
                "color": color, "size": self.np_random.uniform(1, 3)
            })

    def _create_mineral_particles(self, pos, count):
        for _ in range(count):
            direction = self.player_pos - pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            
            vel = direction * self.np_random.uniform(2, 4) + self.np_random.uniform(-0.5, 0.5, size=2)
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": self.np_random.integers(40, 80),
                "color": self.COLOR_MINERAL, "size": self.np_random.uniform(2, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_turrets()
        self._render_asteroids()
        self._render_lasers()
        self._render_mining_beam(self.action_space.sample()[1] == 1) # Visual only
        self._render_particles()
        self._render_player()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size, brightness in self.stars:
            screen_pos = self._world_to_screen(np.array([x, y]))
            if 0 <= screen_pos[0] < self.WIDTH and 0 <= screen_pos[1] < self.HEIGHT:
                color = (brightness, brightness, int(brightness*1.2))
                pygame.draw.circle(self.screen, color, screen_pos, size)

    def _render_turrets(self):
        for turret in self.turrets:
            screen_pos = self._world_to_screen(turret["pos"])
            if 0 <= screen_pos[0] < self.WIDTH and 0 <= screen_pos[1] < self.HEIGHT:
                color = self.COLOR_TURRET_CHARGE if turret["charging"] else self.COLOR_TURRET
                pygame.draw.rect(self.screen, color, (screen_pos[0] - 8, screen_pos[1] - 8, 16, 16))
                pygame.draw.rect(self.screen, self.COLOR_BG, (screen_pos[0] - 5, screen_pos[1] - 5, 10, 10))
                pygame.draw.circle(self.screen, color, screen_pos, 4)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            screen_pos = self._world_to_screen(asteroid["pos"])
            if -50 < screen_pos[0] < self.WIDTH + 50 and -50 < screen_pos[1] < self.HEIGHT + 50:
                rotated_vertices = []
                for x, y in asteroid["vertices"]:
                    rx = x * math.cos(asteroid["angle"]) - y * math.sin(asteroid["angle"])
                    ry = x * math.sin(asteroid["angle"]) + y * math.cos(asteroid["angle"])
                    rotated_vertices.append((screen_pos[0] + rx, screen_pos[1] + ry))
                
                if len(rotated_vertices) > 2:
                    pygame.gfxdraw.aapolygon(self.screen, rotated_vertices, self.COLOR_ASTEROID)
                    pygame.gfxdraw.filled_polygon(self.screen, rotated_vertices, self.COLOR_ASTEROID)

    def _render_lasers(self):
        for laser in self.lasers:
            start_pos = self._world_to_screen(laser["pos"])
            end_pos = self._world_to_screen(laser["pos"] - laser["dir"] * 20)
            pygame.draw.line(self.screen, (255,255,255), start_pos, end_pos, 4)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)

    def _render_mining_beam(self, space_held):
        if not space_held: return
        
        # This is purely visual for _get_observation; actual logic is in step()
        player_screen_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        beam_end_world = self.player_pos + self.last_move_dir * self.BEAM_LENGTH
        beam_end_screen = self._world_to_screen(beam_end_world)

        # Create a semi-transparent surface for the beam
        beam_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(beam_surface, self.COLOR_BEAM, player_screen_pos, beam_end_screen, 8)
        self.screen.blit(beam_surface, (0, 0))


    def _render_player(self):
        pos = (self.WIDTH // 2, self.HEIGHT // 2)
        
        # Thruster
        if np.linalg.norm(self.player_vel) > 0.5:
            angle = math.atan2(self.player_vel[1], self.player_vel[0]) + math.pi
            for i in range(5):
                length = self.np_random.uniform(5, 20)
                p_angle = angle + self.np_random.uniform(-0.3, 0.3)
                start = (pos[0] + 8 * math.cos(angle), pos[1] + 8 * math.sin(angle))
                end = (start[0] + length * math.cos(p_angle), start[1] + length * math.sin(p_angle))
                pygame.draw.line(self.screen, self.COLOR_THRUSTER, start, end, 1+i//2)

        # Body
        angle = math.atan2(self.last_move_dir[1], self.last_move_dir[0])
        p1 = (pos[0] + self.PLAYER_RADIUS * math.cos(angle), pos[1] + self.PLAYER_RADIUS * math.sin(angle))
        p2 = (pos[0] + self.PLAYER_RADIUS * math.cos(angle + 2.3), pos[1] + self.PLAYER_RADIUS * math.sin(angle + 2.3))
        p3 = (pos[0] + self.PLAYER_RADIUS * math.cos(angle - 2.3), pos[1] + self.PLAYER_RADIUS * math.sin(angle - 2.3))
        
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            screen_pos = self._world_to_screen(p["pos"])
            alpha = max(0, min(255, int(255 * (p["life"] / 30))))
            color = (*p["color"], alpha)
            
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (screen_pos[0] - p["size"], screen_pos[1] - p["size"]))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"MINERALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            p1 = (self.WIDTH - 70 + i * 20, 15)
            p2 = (self.WIDTH - 75 + i * 20, 25)
            p3 = (self.WIDTH - 65 + i * 20, 25)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "MISSION COMPLETE" if self.score >= self.WIN_SCORE else "GAME OVER"
        text = self.font_game_over.render(message, True, self.COLOR_UI)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(30) # Run at 30 FPS
        
    env.close()