import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:49:11.617789
# Source Brief: brief_01336.md
# Brief Index: 1336
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    GameEnv: A cyberpunk racing game through a procedural fractal network.

    The agent controls a nano-bot, navigating a glowing track, collecting
    resources, and terraforming paths for speed boosts. It must evade hostile
    security programs whose speed and scanning frequency increase over time.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right
    - `action[1]` (Terraform): 0=Released, 1=Held (Press to terraform)
    - `action[2]` (Boost): 0=Released, 1=Held (Hold to boost)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +100 for reaching the goal.
    - -50 for being caught by a security program.
    - +1.0 for terraforming a path segment.
    - +0.1 for collecting a resource.
    - -0.1 per step inside a security program's scan radius.
    - +0.02 for forward progress towards the goal.
    - -0.01 per step to encourage efficiency.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a nano-bot through a procedural fractal network, collecting resources and "
        "terraforming paths for speed boosts while evading security programs."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Hold shift to boost, and press space to "
        "terraform the track ahead."
    )
    auto_advance = True

    # --- Helper Classes ---
    class Particle:
        def __init__(self, pos, vel, color, radius, lifetime):
            self.pos = pos
            self.vel = vel
            self.color = color
            self.radius = radius
            self.lifetime = lifetime
            self.max_lifetime = lifetime

        def update(self):
            self.pos += self.vel
            self.vel *= 0.95  # Friction
            self.lifetime -= 1
            return self.lifetime > 0

        def draw(self, surface, camera_offset, screen_center):
            screen_pos = self.pos - camera_offset + screen_center
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            current_radius = int(self.radius * (self.lifetime / self.max_lifetime))
            if current_radius > 0:
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (current_radius, current_radius), current_radius)
                surface.blit(temp_surf, (int(screen_pos.x - current_radius), int(screen_pos.y - current_radius)), special_flags=pygame.BLEND_RGBA_ADD)

    class SecurityProgram:
        def __init__(self, pos, speed, scan_radius_base, scan_freq):
            self.pos = pos
            self.base_speed = speed
            self.speed = speed
            self.scan_radius_base = scan_radius_base
            self.scan_freq = scan_freq
            self.scan_radius = scan_radius_base
            self.path_index = 0
            self.path_progress = 0.0
            self.target_node = 1
            self.on_terraformed = False

        def update(self, steps, track_segments):
            # Update dynamic properties
            self.speed = self.base_speed * (1 + (steps // 200) * 0.05)
            current_scan_freq = self.scan_freq * (1 + (steps // 200) * 0.05)
            self.scan_radius = self.scan_radius_base + 20 * math.sin(steps * current_scan_freq)

            # Movement logic
            if self.path_index < len(track_segments):
                segment = track_segments[self.path_index]
                start_node, end_node = segment[0], segment[1]
                segment_vec = end_node - start_node
                segment_len = segment_vec.length()

                if segment_len > 0:
                    self.path_progress += self.speed / segment_len
                    if self.path_progress >= 1.0:
                        self.path_progress = 0.0
                        self.path_index = (self.path_index + 1) % len(track_segments) # Simple loop for now
                
                self.pos = start_node + segment_vec * self.path_progress
                self.on_terraformed = segment[2] == 'terraformed'


        def draw(self, surface, camera_offset, screen_center):
            screen_pos = self.pos - camera_offset + screen_center

            # Draw Scan Radius
            if self.scan_radius > 0:
                scan_alpha = 40 if self.on_terraformed else 25
                radius = int(self.scan_radius)
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*GameEnv.COLOR_SECURITY, scan_alpha))
                pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (*GameEnv.COLOR_SECURITY, scan_alpha + 20))
                surface.blit(temp_surf, (int(screen_pos.x - radius), int(screen_pos.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

            # Draw Body
            radius = 8
            # Glow
            glow_radius = radius + 4
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*GameEnv.COLOR_SECURITY, 80), (glow_radius, glow_radius), glow_radius)
            surface.blit(temp_surf, (int(screen_pos.x - glow_radius), int(screen_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
            # Core
            pygame.gfxdraw.filled_circle(surface, int(screen_pos.x), int(screen_pos.y), radius, GameEnv.COLOR_SECURITY)
            pygame.gfxdraw.aacircle(surface, int(screen_pos.x), int(screen_pos.y), radius, GameEnv.COLOR_SECURITY_ACCENT)

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (5, 0, 15)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_TRACK = (50, 50, 100)
    COLOR_TERRAFORMED = (255, 200, 0)
    COLOR_RESOURCE = (0, 255, 150)
    COLOR_SECURITY = (255, 50, 50)
    COLOR_SECURITY_ACCENT = (255, 150, 150)
    COLOR_TEXT = (220, 220, 255)
    
    PLAYER_RADIUS = 10
    PLAYER_ACCEL = 0.5
    PLAYER_MAX_SPEED = 4.0
    BOOST_MULTIPLIER = 1.8
    RESOURCE_DRAIN_BOOST = 0.2
    TERRAFORM_COST = 25.0
    MAX_RESOURCES = 100.0
    MAX_STEPS = 2000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        
        self.player_pos = None
        self.player_vel = None
        self.player_accel_vec = None
        self.resources = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.last_space_held = None
        self.max_progress = None
        
        self.track_segments = []
        self.track_nodes = []
        self.goal_pos = None
        self.resources_list = []
        self.security_programs = []
        self.particles = []
        
        self.camera_pos = pygame.Vector2(0, 0)
        self.screen_center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_space_held = False
        
        self._generate_track()
        self._spawn_resources()
        
        start_pos = self.track_nodes[0]
        self.player_pos = start_pos.copy()
        self.player_vel = pygame.Vector2(0, 0)
        self.player_accel_vec = pygame.Vector2(0, 0)
        self.resources = 50.0
        self.max_progress = 0.0
        
        self._spawn_security_programs()
        
        self.particles = []
        self.camera_pos = self.player_pos.copy()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        self._update_player_state(shift_held)
        
        reward += self._handle_boost(shift_held)
        reward += self._handle_terraform(space_held)
        
        self._update_security_programs()
        self._update_particles()
        self._update_camera()
        
        reward += self._check_resource_collection()
        reward += self._check_security_scans()
        
        # Reward for progress
        progress = self.player_pos.x
        if progress > self.max_progress:
            reward += (progress - self.max_progress) * 0.02
            self.max_progress = progress
        
        # Small penalty for time
        reward -= 0.01

        terminated = self._check_termination()
        truncated = False
        if terminated:
            self.game_over = True
            if self.win:
                reward += 100.0
            else:
                reward += -50.0
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        self.player_accel_vec.xy = 0, 0
        if movement == 1: self.player_accel_vec.y = -1
        elif movement == 2: self.player_accel_vec.y = 1
        elif movement == 3: self.player_accel_vec.x = -1
        elif movement == 4: self.player_accel_vec.x = 1
        
        if self.player_accel_vec.length() > 0:
            self.player_accel_vec.normalize_ip()

    def _handle_boost(self, shift_held):
        if shift_held and self.resources > 0:
            self.resources = max(0, self.resources - self.RESOURCE_DRAIN_BOOST)
            # Spawn boost particles
            if self.steps % 2 == 0:
                p_vel = -self.player_vel.normalize() * random.uniform(1, 3) + pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
                self._create_particles(self.player_pos.copy(), 1, p_vel, self.COLOR_PLAYER_GLOW, 2, 20)
            return 0
        return 0

    def _handle_terraform(self, space_held):
        if space_held and not self.last_space_held and self.resources >= self.TERRAFORM_COST:
            # Find closest segment in front of player
            best_seg_idx = -1
            min_dist = float('inf')
            
            direction = self.player_vel if self.player_vel.length() > 0 else pygame.Vector2(1,0)

            for i, seg in enumerate(self.track_segments):
                if seg[2] == 'normal':
                    mid_point = (seg[0] + seg[1]) / 2
                    vec_to_mid = mid_point - self.player_pos
                    dist = vec_to_mid.length()
                    
                    if dist < 80 and vec_to_mid.dot(direction) > 0: # In front and close
                        if dist < min_dist:
                            min_dist = dist
                            best_seg_idx = i
            
            if best_seg_idx != -1:
                self.resources -= self.TERRAFORM_COST
                start, end, _ = self.track_segments[best_seg_idx]
                self.track_segments[best_seg_idx] = (start, end, 'terraformed')
                # SFX: TERRAFORM_SUCCESS
                self._create_particles((start+end)/2, 15, pygame.Vector2(0,0), self.COLOR_TERRAFORMED, 4, 30)
                return 1.0
        self.last_space_held = space_held
        return 0

    def _update_player_state(self, boost_active):
        # Apply acceleration
        self.player_vel += self.player_accel_vec * self.PLAYER_ACCEL
        
        # Check for speed boosts
        is_on_terraformed, _ = self._is_on_track(self.player_pos)
        boost_active = boost_active and self.resources > 0
        
        max_speed = self.PLAYER_MAX_SPEED
        if is_on_terraformed:
            max_speed *= 1.25
        if boost_active:
            max_speed *= self.BOOST_MULTIPLIER

        # Cap speed
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)
            
        # Apply friction
        on_track, dist = self._is_on_track(self.player_pos)
        friction = 0.95 if on_track else 0.85
        if is_on_terraformed:
            friction = 0.98
        self.player_vel *= friction

        self.player_pos += self.player_vel

    def _update_security_programs(self):
        for prog in self.security_programs:
            prog.update(self.steps, self.track_segments)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _update_camera(self):
        # Smooth camera follow
        self.camera_pos = self.camera_pos.lerp(self.player_pos, 0.1)

    def _check_resource_collection(self):
        collected_reward = 0
        remaining_resources = []
        for res_pos in self.resources_list:
            if self.player_pos.distance_to(res_pos) < self.PLAYER_RADIUS + 5:
                self.resources = min(self.MAX_RESOURCES, self.resources + 10)
                collected_reward += 0.1
                # SFX: COLLECT_RESOURCE
                self._create_particles(res_pos, 10, pygame.Vector2(0,0), self.COLOR_RESOURCE, 2, 25)
            else:
                remaining_resources.append(res_pos)
        self.resources_list = remaining_resources
        return collected_reward

    def _check_security_scans(self):
        scan_penalty = 0
        for prog in self.security_programs:
            if self.player_pos.distance_to(prog.pos) < prog.scan_radius:
                scan_penalty -= 0.1
        return scan_penalty

    def _check_termination(self):
        # Caught by security
        for prog in self.security_programs:
            if self.player_pos.distance_to(prog.pos) < self.PLAYER_RADIUS + 8: # 8 is prog radius
                # SFX: PLAYER_DEATH
                self._create_particles(self.player_pos, 30, pygame.Vector2(0,0), self.COLOR_PLAYER, 5, 60)
                return True
        
        # Reached goal
        if self.player_pos.distance_to(self.goal_pos) < 50:
            self.win = True
            # SFX: LEVEL_COMPLETE
            self._create_particles(self.player_pos, 50, pygame.Vector2(0,0), self.COLOR_TERRAFORMED, 6, 120)
            return True
            
        return False

    def _generate_track(self):
        self.track_segments = []
        self.track_nodes = []
        
        start_node = pygame.Vector2(100, self.HEIGHT / 2)
        self.track_nodes.append(start_node)

        q = deque([(start_node, pygame.Vector2(1, 0), 5)]) # pos, dir, depth

        while q:
            curr_pos, curr_dir, depth = q.popleft()
            if depth <= 0 or curr_pos.x > self.WIDTH * 4:
                continue

            length = self.np_random.uniform(80, 150)
            end_pos = curr_pos + curr_dir * length
            
            self.track_segments.append([curr_pos, end_pos, 'normal'])
            if end_pos not in self.track_nodes:
                self.track_nodes.append(end_pos)

            # Main path
            main_angle = self.np_random.uniform(-0.3, 0.3) # radians
            q.append((end_pos, curr_dir.rotate_rad(main_angle), depth - 1))

            # Branching
            if self.np_random.random() < 0.4 and depth > 1:
                branch_angle = self.np_random.uniform(0.6, 1.2) * self.np_random.choice([-1, 1])
                q.append((end_pos, curr_dir.rotate_rad(branch_angle), depth - 2))
        
        # Set goal
        self.goal_pos = max(self.track_nodes, key=lambda p: p.x)

    def _spawn_resources(self):
        self.resources_list = []
        for start, end, _ in self.track_segments:
            if self.np_random.random() < 0.3:
                pos = start.lerp(end, self.np_random.random())
                self.resources_list.append(pos)

    def _spawn_security_programs(self):
        self.security_programs = []
        for i in range(3):
            path_idx = self.np_random.integers(0, len(self.track_segments))
            start_node = self.track_segments[path_idx][0]
            prog = self.SecurityProgram(
                pos=start_node.copy(),
                speed=self.np_random.uniform(1.0, 1.5),
                scan_radius_base=self.np_random.uniform(50, 80),
                scan_freq=self.np_random.uniform(0.05, 0.1)
            )
            prog.path_index = path_idx
            self.security_programs.append(prog)
            
    def _is_on_track(self, pos):
        min_dist = float('inf')
        on_terraformed = False
        for start, end, type in self.track_segments:
            dist = self._point_segment_distance(pos, start, end)
            if dist < min_dist:
                min_dist = dist
                on_terraformed = (type == 'terraformed')
        return min_dist < 20, on_terraformed

    def _point_segment_distance(self, p, a, b):
        if a == b: return p.distance_to(a)
        l2 = a.distance_squared_to(b)
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        projection = a + t * (b - a)
        return p.distance_to(projection)

    def _create_particles(self, pos, num, base_vel, color, radius, lifetime):
        for _ in range(num):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed + base_vel
            self.particles.append(self.Particle(pos.copy(), vel, color, radius, int(lifetime * random.uniform(0.8, 1.2))))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Goal
        goal_screen_pos = self.goal_pos - self.camera_pos + self.screen_center
        if 0 < goal_screen_pos.x < self.WIDTH and 0 < goal_screen_pos.y < self.HEIGHT:
            radius = 50
            alpha = 50 + 30 * math.sin(self.steps * 0.1)
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*self.COLOR_TERRAFORMED, int(alpha)))
            self.screen.blit(temp_surf, (int(goal_screen_pos.x-radius), int(goal_screen_pos.y-radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Render Track
        for start, end, type in self.track_segments:
            color = self.COLOR_TERRAFORMED if type == 'terraformed' else self.COLOR_TRACK
            s_pos = start - self.camera_pos + self.screen_center
            e_pos = end - self.camera_pos + self.screen_center
            pygame.draw.line(self.screen, color, (int(s_pos.x), int(s_pos.y)), (int(e_pos.x), int(e_pos.y)), 3)

        # Render Resources
        for res_pos in self.resources_list:
            screen_pos = res_pos - self.camera_pos + self.screen_center
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), 5, self.COLOR_RESOURCE)
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), 5, self.COLOR_RESOURCE)

        # Render Particles
        for p in self.particles:
            p.draw(self.screen, self.camera_pos, self.screen_center)

        # Render Security Programs
        for prog in self.security_programs:
            prog.draw(self.screen, self.camera_pos, self.screen_center)
            
        # Render Player
        player_screen_pos = self.player_pos - self.camera_pos + self.screen_center
        # Glow
        glow_radius = int(self.PLAYER_RADIUS * 1.8)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER_GLOW, 100), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (int(player_screen_pos.x - glow_radius), int(player_screen_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        # Body
        pygame.gfxdraw.filled_circle(self.screen, int(player_screen_pos.x), int(player_screen_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(player_screen_pos.x), int(player_screen_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
        # Direction indicator
        if self.player_vel.length() > 0.1:
            p2 = player_screen_pos + self.player_vel.normalize() * (self.PLAYER_RADIUS + 2)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, player_screen_pos, p2, 3)

    def _render_ui(self):
        # Resource Bar
        res_bar_width = 150
        res_bar_height = 20
        res_ratio = self.resources / self.MAX_RESOURCES
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (10, 10, res_bar_width, res_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_RESOURCE, (10, 10, int(res_bar_width * res_ratio), res_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 10, res_bar_width, res_bar_height), 1)
        res_text = self.font.render("RESOURCES", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (15, 12))

        # Score / Time
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "win": self.win
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run with the `dummy` video driver, so we unset it.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Fractal Runner")
    clock = pygame.time.Clock()

    while not done:
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    print(f"Game Over. Final Info: {info}")
    env.close()